import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv, GATv2Conv, dense_diff_pool, DenseGCNConv, DenseSAGEConv
from torch_geometric.utils import dropout_edge, to_dense_batch, to_dense_adj
from model_help import *
from torch import nn


class NeighborEmbedding(nn.Module):
    def __init__(self, num_embeddings, out_channels=128, dropout=0.5, cached=True, bias=True, lamda=0.8, share=True):
        super(NeighborEmbedding, self).__init__()
        self.shutcut = nn.Linear(in_features=num_embeddings, out_features=out_channels)
        self.embedding_sim = Embedding_sim(in_channels=num_embeddings, out_channels=out_channels,
                             cached=cached, bias=bias, lamda=lamda, share=share)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels

    def forward(self, x, edge, embedding):
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index
        embedding = self.embedding_sim(embedding, edge_index=edge_index)
        embedding = self.dropout(embedding)
        x = F.embedding(x, embedding)
        x = F.normalize(x)
        return x

class InteractionEmbedding(nn.Module):
    def __init__(self, n_drug, n_disease, embedding_dim, dropout=0.5):
        super(InteractionEmbedding, self).__init__()
        self.drug_project = nn.Linear(n_drug, embedding_dim, bias=False)
        self.disease_project = nn.Linear(n_disease, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim

    def forward(self, association_pairs, drug_embedding, disease_embedding):
        drug_embedding = torch.diag(torch.ones(drug_embedding.shape[0], device=drug_embedding.device))
        disease_embedding = torch.diag(torch.ones(disease_embedding.shape[0], device=disease_embedding.device))
        drug_embedding = self.drug_project(drug_embedding)
        disease_embedding = self.disease_project(disease_embedding)
        drug_embedding = F.embedding(association_pairs[0, :], drug_embedding)
        disease_embedding = F.embedding(association_pairs[1, :], disease_embedding)
        associations = drug_embedding * disease_embedding
        associations = F.normalize(associations)
        associations = self.dropout(associations)
        return associations

class InteractionDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dims=(256, 64), out_channels=1, dropout=0.5):
        super(InteractionDecoder, self).__init__()
        decoder = []
        in_dims = [in_channels] + list(hidden_dims)
        out_dims = hidden_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            decoder.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.Dropout(dropout))
        decoder.append(nn.Linear(hidden_dims[-1], out_channels))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


import torch.nn.functional as F

def pad_to_match(tensor, target_size, dim=0):
    if tensor.size(dim) < target_size:
        padding = [0, 0] * (tensor.dim() - dim - 1) + [0, target_size - tensor.size(dim)]
        tensor = F.pad(tensor, padding, "constant", 0)
    return tensor

class SAGAN(torch.nn.Module):
    def __init__(self, dataset, gconv=GATConv, latent_dim=[128, 64, 32], k=30,
                 dropout_n=0.4, dropout_e=0.1, heads=[3, 2, 2], force_undirected=False, num_classes=1, sim_data=None):
        super(SAGAN, self).__init__()

        self.dropout_n = dropout_n
        self.dropout_e = dropout_e
        self.force_undirected = force_undirected
        self.heads = heads
        
        self.drug_neighbor_encoder = NeighborEmbedding(num_embeddings=dataset.n_drug,
                                                       out_channels=32,
                                                       dropout=0.5, lamda=0.8)
        self.disease_neighbor_encoder = NeighborEmbedding(num_embeddings=dataset.n_disease,
                                                          out_channels=32,
                                                          dropout=0.5, lamda=0.8)
        # self.interaction_encoder = InteractionEmbedding(n_drug=dataset.n_drug, n_disease=dataset.n_disease,
        #                                                embedding_dim=64, dropout=dropout_e)  
        # merged_dim = self.disease_neighbor_encoder.output_dim + self.drug_neighbor_encoder.output_dim + self.interaction_encoder.output_dim
        
        self.conv1 = GATv2Conv(dataset.num_features, latent_dim[0], heads=heads[0], dropout=dropout_e)
        self.conv2 = GATv2Conv(latent_dim[0] * heads[0], latent_dim[1], heads=heads[1], dropout=dropout_e)
        self.conv3 = GATv2Conv(latent_dim[1] * heads[1], latent_dim[2], heads=heads[2], concat=False, dropout=dropout_e)

        self.bn1 = BatchNorm1d(latent_dim[0] * heads[0])
        self.bn2 = BatchNorm1d(latent_dim[1] * heads[1])
        self.bn3 = BatchNorm1d(latent_dim[2] * heads[2])

        self.res1 = torch.nn.Linear(latent_dim[0] * heads[0], latent_dim[1] * heads[1])
        self.res2 = torch.nn.Linear(latent_dim[1] * heads[1], latent_dim[2])

        fusion_input_dim = 32 + 32 + latent_dim[2]
        self.attn_fusion = nn.Linear(fusion_input_dim, latent_dim[2])

        total_latent_dim = latent_dim[0] * heads[0] + latent_dim[1] * heads[1] + latent_dim[2]
        self.diffpool_embed = Linear(total_latent_dim, k)
        self.diffpool_gcn = DenseGCNConv(total_latent_dim, total_latent_dim)

        self.fc_weight_subgraph = nn.Parameter(torch.ones(1))
        self.fc_weight_global = nn.Parameter(torch.ones(1))

        self.decoder = InteractionDecoder(in_channels=total_latent_dim * k + 64, hidden_dims=(64, 32), dropout=0.5)
        self.lin1 = Linear(total_latent_dim * k + 64, 128)
        self.lin2 = Linear(128, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.diffpool_gcn.reset_parameters()
        self.diffpool_embed.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, sim_data, return_attention_weights=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_e, force_undirected=self.force_undirected, training=self.training)

        drug_neighbor_embedding = self.drug_neighbor_encoder(
            x=data.interaction_pairs1, edge=(sim_data.drug_edge), embedding=sim_data.drug_embedding
        )
        disease_neighbor_embedding = self.disease_neighbor_encoder(
            x=data.interaction_pairs2, edge=(sim_data.disease_edge), embedding=sim_data.disease_embedding
        )

        x1 = self.bn1(F.relu(self.conv1(x, edge_index)))
        x2 = self.bn2(F.relu(self.conv2(x1, edge_index)))
        x3, attn_weights = self.conv3(x2, edge_index, return_attention_weights=True)
        x = torch.cat((x1, x2, x3), dim=1)

        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch)

        s = self.diffpool_embed(x_dense)
        x_pooled, adj_pooled, link_loss, ent_loss = dense_diff_pool(x_dense, adj_dense, s, mask=mask)

        x_pooled = F.relu(self.diffpool_gcn(x_pooled, adj_pooled))
        x_pooled = x_pooled.view(x_pooled.size(0), -1)

        global_embedding = torch.cat([drug_neighbor_embedding, x_pooled, disease_neighbor_embedding], dim=-1)
        
        x = F.relu(self.lin1(global_embedding))
        x = F.dropout(x, p=self.dropout_n, training=self.training)
        x = self.lin2(x)

        if return_attention_weights:
            return x[:, 0], attn_weights
        else:
            return x[:, 0]
