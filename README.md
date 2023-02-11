# Implementation of GNN
PyTorch implementation of Graph Convolutional Networks (GCN) and Deep Graph CNN (DGCNN)

## Reference Papers
GCN is based on the paper presented in ICLR 2017:
- [Semi-supervised classification with graph convolutional networks](https://openreview.net/pdf?id=SJU4ayYgl)

DGCNN is based on the paper presented in AAAI 2018:
- [An End-to-End Deep Learning Architecture for Graph Classification](https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf)

## Requirements
- PyTorch
- PyTorch Geometric

## Datasets
This implementation uses the Lab dataset which includes two classes:
- Benign
- Mirai

## Results
The results of the implementation are shown in the following table:

| Detector | Training Accuracy | Testing Accuracy |
| -------- | ----------------- | ---------------- |
| GCN      | 99.43%            | 99.36%           |
| DGCN     | 99.56%            | 99.53%           |