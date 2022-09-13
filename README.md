# Implementation-GNN
PyTorch implementation GCN and GDCNN

## Paper
GCN based on ICLR 2017
- [Semi-supervised classification with graph convolutional networks](https://openreview.net/pdf?id=SJU4ayYgl)

DGCNN based on AAAI 2018
- [An End-to-End Deep Learning Architecture for Graph Classification](https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf)

# Requirements
- Pytorch
- PyTorch Geometric

# Datasets
Lab dataset
- benign
- mirai

# Result
| Detector | Train | Test |
| -------- | -------- | -------- |
| GCN | 99.43% | 99.36% |
| DGCN | 99.56% | 99.53% |