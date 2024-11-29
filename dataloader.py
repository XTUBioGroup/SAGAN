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

import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))
        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


def neighbors(fringe, A):
    return set(A[list(fringe)].indices)

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels, h, drug_embedding, disease_embedding):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h

        self.n_drug = drug_embedding.shape[0]
        self.n_disease = disease_embedding.shape[0]
 

    def len(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction((i, j), self.Arow, self.Acol, g_label, self.h)
        data = construct_pyg_graph(*tmp[0:6])
        data.interaction_pairs1 = torch.tensor(i, dtype=torch.long)
        data.interaction_pairs2 = torch.tensor(j, dtype=torch.long)
        return data

class MyDataset(InMemoryDataset):
    def __init__(self, root, A, links, labels, hop = 2, drug_names = None, disease_names = None):
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.hop = hop
        self.drug_names = drug_names
        self.disease_names = disease_names
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'data.pt'
        return [name]

    def process(self):
        # Extract enclosing subgraphs and save to disk
        data_list = links2subgraphs(self.Arow, self.Acol, self.links, self.labels, self.hop, self.drug_names, self.disease_names)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del data_list


def subgraph_extraction(ind, Arow, Acol, label=1, h=1):
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    for dist in range(1, h + 1):
        if len(u_fringe) == 0 or len(v_fringe) == 0:
            break

        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited

        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)

        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    subgraph[0, 0] = 0
    u, v, r = ssp.find(subgraph)
    v += len(u_nodes)
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    max_node_label = 2 * 8 + 1

    return u, v, r, node_labels, max_node_label, label


def construct_pyg_graph(u, v, r, node_labels, max_node_label, y):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_attr=edge_type, y=y)

    return data

def links2subgraphs(Arow, Acol, links, labels, hop, drug_names, disease_names):
    # extract enclosing subgraphs
    print('Enclosing subgraph extraction begins...')
    g_list = []

    # 确保 drug_names 和 disease_names 是可直接索引的一维数组
    drug_names = drug_names.flatten()
    disease_names = disease_names.flatten()

    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(
        subgraph_extraction,
        [
            ((i, j), Arow, Acol, g_label, hop, drug_names, disease_names)
            for i, j, g_label in zip(links[0], links[1], labels)
        ]
    )
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready():
            break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    end = time.time()
    print("Time elapsed for subgraph extraction: {}s".format(end - start))
    print("Transforming to pytorch_geometric graphs...")
    g_list = []
    pbar = tqdm(total=len(results))
    while results:
        tmp = results.pop()
        g_list.append(construct_pyg_graph_with_names(*tmp))
        pbar.update(1)
    pbar.close()
    end2 = time.time()
    print("Time elapsed for transforming to pytorch_geometric graphs: {}s".format(end2 - end))

    return g_list

def build_similarity_graph(sim, num_neighbor=5):
    if isinstance(sim, list):
        sim = np.array(sim)
    if num_neighbor > sim.shape[0] or num_neighbor < 0:
        num_neighbor = sim.shape[0]

    neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]
    row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
    col_index = neighbor.reshape(-1)
    
    edge_index = torch.from_numpy(np.array([row_index, col_index]).astype(int))
    
    values = torch.from_numpy(sim[row_index, col_index]).float()
    
    return edge_index, values, (sim.shape[0], sim.shape[0])

def row_normalize_matrix(matrix):
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums[:, np.newaxis]

class SData:
    def __init__(self, drug_embedding, disease_embedding, drug_edge, disease_edge):
        self.drug_embedding = drug_embedding
        self.disease_embedding = disease_embedding
        self.drug_edge = drug_edge
        self.disease_edge = disease_edge

    def get_drug_embedding_length(self):
        return self.drug_embedding.shape[0]

    def get_disease_embedding_length(self):
        return self.disease_embedding.shape[0]

    def get_drug_edge_length(self):
        return self.drug_edge.shape[0]

    def get_disease_edge_length(self):
        return self.disease_edge.shape[0]
