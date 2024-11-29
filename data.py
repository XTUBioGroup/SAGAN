from dataloader import *
import pandas as pd
import numpy as np
import io
import warnings
import os
import torch
import time
import networkx as nx
import scipy.sparse as ssp
import multiprocessing as mp
from tqdm import tqdm
from scipy import io
from torch_geometric.data import Data, InMemoryDataset, Dataset
from sklearn.model_selection import KFold
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def extract_subgraph(split_data, k, args):
    (
        adj_train, train_labels, train_u_indices, train_v_indices,
        test_labels, test_u_indices, test_v_indices, drug_embedding, disease_embedding
    ) = split_data
    val_test_appendix = str(k) + '_kfold'
    data_combo = (args.data_name, val_test_appendix)  

    train_indices = (train_u_indices, train_v_indices)
    test_indices = (test_u_indices, test_v_indices)

    train_file_path = 'data/{}/{}/train'.format(*data_combo)
    train_graph = MyDynamicDataset(train_file_path, adj_train, train_indices, train_labels, args.hop, drug_embedding, disease_embedding)

    test_file_path = 'data/{}/{}/test'.format(*data_combo)
    test_graph = MyDynamicDataset(test_file_path, adj_train, test_indices, test_labels, args.hop, drug_embedding, disease_embedding)

    return train_graph, test_graph


def load_k_fold(data_name, seed, pos_neg_ratio=(1, 1)):
    root_path = os.path.dirname(os.path.abspath(__file__))
    if data_name == 'lrssl':
        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.txt')
        matrix = pd.read_table(path, index_col=0).values
        drug_sim = pd.read_csv(os.path.join(root_path, 'drug_data/LRSSL/lrssl_simmat_dc_chemical.txt'), 
                              sep="\t", index_col=0).values
        disease_sim = pd.read_csv(os.path.join(root_path, 'drug_data/LRSSL/lrssl_simmat_dg.txt'),
                                 sep="\t", index_col=0).values
    elif data_name in ['Gdataset', 'Cdataset', 'Fdataset']:
        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.mat')
        data = io.loadmat(path)
        matrix = data['didr'].T
        drug_sim = data['drug'].astype(np.float64)
        disease_sim = data['disease'].astype(np.float64)
    else:
        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.csv')
        data = pd.read_csv(path, header=None)
        matrix = data.values.T
        drug_sim = data['drug_sim'].values  
        disease_sim = data['disease_sim'].values

    drug_sim = row_normalize_matrix(drug_sim)
    disease_sim = row_normalize_matrix(disease_sim)

    drug_num, disease_num = matrix.shape[0], matrix.shape[1]
    drug_id, disease_id = np.nonzero(matrix)

    num_len = int(np.ceil(len(drug_id) * 1))
    drug_id, disease_id = drug_id[0: num_len], disease_id[0: num_len]

    neutral_flag = 0
    labels = np.full((drug_num, disease_num), neutral_flag, dtype=np.int32)
    observed_labels = [1] * len(drug_id)
    labels[drug_id, disease_id] = np.array(observed_labels)
    labels = labels.reshape([-1])

    num_train = int(np.ceil(0.9 * len(drug_id)))
    num_test = int(np.ceil(0.1 * len(drug_id)))
    print("num_train, num_test's ratio is", 0.9, 0.1)
    print("num_train {}".format(num_train),
          "num_test {}".format(num_test))

    neg_drug_idx, neg_disease_idx = np.where(matrix == 0)
    neg_pairs = np.array([[dr, di] for dr, di in zip(neg_drug_idx, neg_disease_idx)])

    np.random.seed(seed)
    np.random.shuffle(neg_pairs)

    pos_pairs = np.array([[dr, di] for dr, di in zip(drug_id, disease_id)])
    pos_idx = np.array([dr * disease_num + di for dr, di in pos_pairs])

    split_data_dict = {}
    count = 0
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train_data, test_data in kfold.split(pos_idx):
        idx_pos_train = np.array(pos_idx)[np.array(train_data)]
        num_neg_samples = int(len(idx_pos_train) * pos_neg_ratio[1] / pos_neg_ratio[0])

        idx_neg_train = []
        for neg in neg_pairs:
            if len(idx_neg_train) >= num_neg_samples:
                break
            neg_idx = neg[0] * disease_num + neg[1]
            if neg_idx not in idx_pos_train:
                idx_neg_train.append(neg_idx)
        idx_neg_train = np.array(idx_neg_train)

        idx_train = np.concatenate([idx_pos_train, idx_neg_train], axis=0)

        pairs_pos_train = pos_pairs[np.array(train_data)]
        pairs_neg_train = neg_pairs[:num_neg_samples]
        pairs_train = np.concatenate([pairs_pos_train, pairs_neg_train], axis=0)

        idx_pos_test = np.array(pos_idx)[np.array(test_data)]
        num_neg_test_samples = len(idx_pos_test)
        idx_neg_test = []
        for neg in neg_pairs[num_neg_samples:]:
            if len(idx_neg_test) >= num_neg_test_samples:
                break
            neg_idx = neg[0] * disease_num + neg[1]
            if neg_idx not in idx_pos_test:
                idx_neg_test.append(neg_idx)
        idx_neg_test = np.array(idx_neg_test)

        idx_test = np.concatenate([idx_pos_test, idx_neg_test], axis=0)

        pairs_pos_test = pos_pairs[np.array(test_data)]
        pairs_neg_test = neg_pairs[num_neg_samples:num_neg_samples + num_neg_test_samples]
        pairs_test = np.concatenate([pairs_pos_test, pairs_neg_test], axis=0)

        rand_idx = list(range(len(idx_train)))
        np.random.seed(42)
        np.random.shuffle(rand_idx)
        idx_train = idx_train[rand_idx]
        pairs_train = pairs_train[rand_idx]

        u_train_idx, v_train_idx = pairs_train.transpose()
        u_test_idx, v_test_idx = pairs_test.transpose()

        train_labels = labels[idx_train]
        test_labels = labels[idx_test]

        rating_mx_train = np.zeros(drug_num * disease_num, dtype=np.float32)
        rating_mx_train[idx_train] = labels[idx_train]
        rating_mx_train = ssp.csr_matrix(rating_mx_train.reshape(drug_num, disease_num))

        split_data_dict[count] = [rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, drug_sim, disease_sim]

        count += 1

    return split_data_dict, drug_sim, disease_sim



if __name__ == '__main__':
    split_data_dict = load_k_fold('Fdataset', 1, pos_neg_ratio=(1, 1))
