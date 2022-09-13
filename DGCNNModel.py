import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import remove_self_loops

class DGCNN(nn.Module):
  def __init__(self, num_features, num_classes):
    super(DGCNN, self).__init__()
    print('\033[1;32m===> DGCNN Model......\033[0m')
    self.conv1 = GCNConv(num_features, 32)
    self.conv2 = GCNConv(32, 32)
    self.conv3 = GCNConv(32, 32)
    self.conv4 = GCNConv(32, 1)
    self.conv5 = Conv1d(1, 16, 97, 97)
    self.conv6 = Conv1d(16, 32, 5, 1)
    self.pool = MaxPool1d(2, 2)
    self.classifier_1 = Linear(352, 128)
    # self.classifier_1 = Linear(512, 128)  # k = 40 -> (40 - 30) / 10 * 160 + 352
    # self.classifier_1 = Linear(576, 128)  # k = 45
    # self.classifier_1 = Linear(672, 128)  # k = 50
    # self.classifier_1 = Linear(832, 128)  # k = 60
    self.classifier_1 = Linear(992, 128)  # k = 70
    # self.classifier_1 = Linear(1152, 128)  # k = 80
    # self.classifier_1 = Linear(1312, 128)  # k = 90
    # self.classifier_1 = Linear(1472, 128)  # k = 100
    self.drop_out = Dropout(0.5)
    self.classifier_2 = Linear(128, num_classes)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x, edges, batch):
    # x, edge_index, batch = data.x, data.edge_index, data.batch
    edges, _ = remove_self_loops(edges)

    x_1 = torch.tanh(self.conv1(x, edges))
    x_2 = torch.tanh(self.conv2(x_1, edges))
    x_3 = torch.tanh(self.conv3(x_2, edges))
    x_4 = torch.tanh(self.conv4(x_3, edges))
    x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
    x = global_sort_pool(x, batch, k=70)
    x = x.view(x.size(0), 1, x.size(-1))
    x = self.relu(self.conv5(x))
    x = self.pool(x)
    x = self.relu(self.conv6(x))
    x = x.view(x.size(0), -1)
    x = self.relu(self.classifier_1(x))
    out = self.relu(self.classifier_2(x))
    # out = self.drop_out(out)
    # classes = F.log_softmax(self.classifier_2(out), dim=-1)

    return out


# class DGCNN(nn.Module):
#   def __init__(self, num_features, num_classes):
#     super(DGCNN, self).__init__()

#     self.conv1 = GCNConv(num_features, 32)
#     self.conv2 = GCNConv(32, 32)
#     self.conv3 = GCNConv(32, 32)
#     self.conv4 = GCNConv(32, 1)
#     self.conv5 = Conv1d(1, 16, 97, 97)
#     self.conv6 = Conv1d(16, 32, 5, 1)
#     self.pool = MaxPool1d(2, 2)
#     self.classifier_1 = Linear(352, 128)
#     self.drop_out = Dropout(0.5)
#     self.classifier_2 = Linear(128, num_classes)
#     self.relu = nn.ReLU(inplace=True)

#   def forward(self, data):
#     x, edge_index, batch = data.x, data.edge_index, data.batch
#     edge_index, _ = remove_self_loops(edge_index)

#     x_1 = torch.tanh(self.conv1(x, edge_index))
#     x_2 = torch.tanh(self.conv2(x_1, edge_index))
#     x_3 = torch.tanh(self.conv3(x_2, edge_index))
#     x_4 = torch.tanh(self.conv4(x_3, edge_index))
#     x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
#     x = global_sort_pool(x, batch, k=30)
#     x = x.view(x.size(0), 1, x.size(-1))
#     x = self.relu(self.conv5(x))
#     x = self.pool(x)
#     x = self.relu(self.conv6(x))
#     x = x.view(x.size(0), -1)
#     out = self.relu(self.classifier_1(x))
#     out = self.drop_out(out)
#     classes = F.log_softmax(self.classifier_2(out), dim=-1)

#     return classes