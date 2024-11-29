# SAGAN
A Subgraph-Aware Graph Attention Network for Drug repositioning

# Requirments

```
python==3.8.10
torch==2.0.1+cu117
torch-geometric==2.6.1
torch-scatter==2.1.2+pt20cu118
torch-sparse==0.6.18+pt20cu118
pytorch_lightning==1.9.0
labml==0.4.168
In our experiment setting: cuda11.7
```
Note: If there is a problem with the torch-sparse installation, please use the link 
[https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu117.html](https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu117.html) to download the appropriate torch-sparse version.

# Datasets
- Fdataset
- Cdataset
- LRSSL

# How to use
```
run main.py
```
