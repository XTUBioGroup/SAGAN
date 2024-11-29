import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn import metrics


class BaseModel(pl.LightningModule):
    DATASET_TYPE: None

    def __init__(self):
        super(BaseModel, self).__init__()

    def select_topk(self, data, k=-1):
        if k is None or k <= 0:
            return data
        assert k <= data.shape[1]
        val, col = torch.topk(data, k=k)
        col = col.reshape(-1)
        row = torch.ones(1, k, dtype=torch.int) * torch.arange(data.shape[0]).view(-1, 1)
        row = row.view(-1).to(device=data.device)
        new_data = torch.zeros_like(data)
        new_data[row, col] = data[row, col]
        return new_data

    def merge_neighbor_feature(self, sims, features, k=5):
        assert sims.shape[0] == features.shape[0] and sims.shape[1] == sims.shape[0]
        if k < 0:
            k = sims.shape[1]
        N = features.shape[0]
        value, idx = torch.topk(sims, dim=1, k=k)
        col = idx.reshape(-1)
        features = features[col].view(N, k, -1) * value.view(N, k, 1)
        features = features.sum(dim=1)
        features = features / value.sum(dim=1).view(N, 1)
        return features

    def neighbor_smooth(self, sims, features, replace_rate=0.2):
        merged_u = self.merge_neighbor_feature(sims, features)
        mask = torch.rand(merged_u.shape[0], device=sims.device)
        mask = torch.floor(mask + replace_rate).type(torch.bool)
        new_features = torch.where(mask, merged_u, features)
        return new_features

    def laplacian_matrix(self, S):
        x = torch.sum(S, dim=0)
        y = torch.sum(S, dim=1)
        L = 0.5 * (torch.diag(x + y) - (S + S.T))
        return L

    def graph_loss_fn(self, x, edge, topk=None, cache_name=None, reduction="mean"):
        if not hasattr(self, f"_{cache_name}"):
            adj = torch.sparse_coo_tensor(*edge).to_dense()
            adj = adj - torch.diag(torch.diag(adj))
            adj = self.select_topk(adj, k=topk)
            la = self.laplacian_matrix(adj)
            if cache_name:
                self.register_buffer(f"_{cache_name}", la)
        else:
            la = getattr(self, f"_{cache_name}")
            assert la.shape == edge[2]

        graph_loss = torch.trace(x.T @ la @ x)
        graph_loss = graph_loss / (x.shape[0] ** 2) if reduction == "mean" else graph_loss
        return graph_loss

    def mse_loss_fn(self, predict, label, pos_weight):
        predict = predict.view(-1)
        label = label.view(-1)
        pos_mask = label > 0
        loss = F.mse_loss(predict, label, reduction="none")
        loss_pos = loss[pos_mask].mean()
        loss_neg = loss[~pos_mask].mean()
        loss_mse = loss_pos * pos_weight + loss_neg
        return {"loss_mse": loss_mse,
                "loss_mse_pos": loss_pos,
                "loss_mse_neg": loss_neg,
                "loss": loss_mse}

    def bce_loss_fn(self, predict, label, pos_weight):
        predict = predict.view(-1)
        label = label.view(-1)
        weight = pos_weight * label + 1 - label
        loss = F.binary_cross_entropy(input=predict, target=label, weight=weight)
        return {"loss_bce": loss,
                "loss": loss}

    def focal_loss_fn(self, predict, label, alpha, gamma):
        predict = predict.view(-1)
        label = label.view(-1)
        ce_loss = F.binary_cross_entropy(
            predict, label, reduction="none"
        )
        p_t = predict * label + (1 - predict) * (1 - label)
        loss = ce_loss * ((1 - p_t) ** gamma)
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        focal_loss = (alpha_t * loss).mean()
        return {"loss_focal": focal_loss,
                "loss": focal_loss}

    def rank_loss_fn(self, predict, label, margin=0.8, reduction='mean'):
        predict = predict.view(-1)
        label = label.view(-1)
        pos_mask = label > 0
        pos = predict[pos_mask]
        neg = predict[~pos_mask]
        neg_mask = torch.randint(0, neg.shape[0], (pos.shape[0],), device=label.device)
        neg = neg[neg_mask]

        rank_loss = F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos),
                                          margin=margin, reduction=reduction)
        return {"loss_rank": rank_loss,
                "loss": rank_loss}

    def get_epoch_auroc_aupr(self, outputs):
        predict = [output["predict"].detach() for output in outputs]
        label = [output["label"] for output in outputs]
        predict = torch.cat(predict).cpu().view(-1)
        label = torch.cat(label).cpu().view(-1)
        aupr = metrics.average_precision_score(y_true=label, y_score=predict)
        auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
        return auroc, aupr

    def get_epoch_loss(self, outputs):
        loss_keys = [key for key in outputs[0] if key.startswith("loss")]
        loss_info = {key: [output[key].detach().cpu() for output in outputs if not torch.isnan(output[key])] for key in
                     loss_keys}
        loss_info = {key: sum(value) / len(value) for key, value in loss_info.items()}
        return loss_info

    def training_epoch_end(self, outputs):
        stage = "train"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):

        stage = "val"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict

import numpy as np
import torch
from torch import nn

from labml import lab, monit, tracker, experiment
from labml.configs import BaseConfigs, option, calculate
from labml.utils import download
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_nn.graphs.gat import GraphAttentionLayer
from labml_nn.optimizers.configs import OptimizerConfigs


class GAT(Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=False, dropout=dropout)
        self.activation = nn.ELU()
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat)



def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape, device=adj_t.device)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index, None


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = True, add_self_loops: bool = False,
                 bias: bool = True, **kwargs):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        if self.cached:
            if not hasattr(self, "cached_adj"):
                edge_index, edge_weight = gcn_norm(
                    edge_index, self.add_self_loops)
                self.register_buffer("cached_adj", edge_index)
            edge_index = self.cached_adj
        else:
            edge_index, _ = gcn_norm(edge_index, self.add_self_loops)
        x = torch.matmul(x, self.weight)

        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class Embedding_sim(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, lamda=0.8, share=True, **kwargs):
        super(Embedding_sim, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        # self.gat = GAT(in_channels, n_hidden=256, n_classes=out_channels, n_heads=8, dropout=0.5)
        self.gat = GAT(in_channels, n_hidden=64, n_classes=out_channels, n_heads=2, dropout=0.5)
        self.gcn = GCNConv(in_channels=in_channels, out_channels=out_channels, cached=cached,
                           add_self_loops=add_self_loops, bias=bias)

        self.register_buffer("alpha", torch.tensor(lamda))
        self.register_buffer("beta", torch.tensor(1-lamda))
        self.reset_parameters()
        if share:
            self.gat.weight = self.gcn.weight

    def reset_parameters(self):
        self.gcn.reset_parameters()

    def forward(self, x, edge_index):
        adj_matrix = edge_index.to_dense()
        m = 1
        adj_matrix = torch.unsqueeze(adj_matrix, dim=2).repeat(1, 1, m)
        x1 = self.gcn(x, edge_index)
        x2=self.gat(x, adj_matrix)
        x = self.beta*F.relu(x1)+self.alpha*F.relu(x2)
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
