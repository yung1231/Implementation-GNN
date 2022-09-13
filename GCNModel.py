import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
  def __init__(self, num_features, num_classes):
    super(GCN, self).__init__()
    print('\033[1;32m===> GCN Model......\033[0m')
    # torch.manual_seed(12345)
    self.conv1 = GCNConv(num_features, 128)
    self.conv2 = GCNConv(128, 64)
    self.conv3 = GCNConv(64, 32)
    self.conv4 = GCNConv(32, 16)
    self.pred = Linear(16, num_classes)

  def forward(self, x, edges, batch):
    #! Semantics-preserving Reinforcement Learning Attack Against Graph Neural Networks for Malware Detection：GCN
    x = self.conv1(x, edges)
    x = F.relu(x)
    x = self.conv2(x, edges)
    x = F.relu(x)
    x = self.conv3(x, edges)
    x = F.relu(x)
    x = self.conv4(x, edges)
    x = F.relu(x)
    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
    # x = self.fc1(x)
    x = self.pred(x)

    return x



# class GCN(torch.nn.Module):
#   def __init__(self, num_features, hidden_channels, num_classes):
#     super(GCN, self).__init__()
#     # torch.manual_seed(12345)
#     self.conv1 = GCNConv(num_features, hidden_channels)
#     self.conv2 = GCNConv(hidden_channels, hidden_channels)
#     self.conv3 = GCNConv(hidden_channels, hidden_channels*2)
#     self.conv4 = GCNConv(hidden_channels*2, hidden_channels)
#     self.lin1 = Linear(hidden_channels, hidden_channels/2)
#     self.lin2 = Linear(hidden_channels/2, hidden_channels/4)
#     self.pred = Linear(hidden_channels/4, num_classes)

#   def forward(self, x, edges, batch):
#     x = F.relu(self.conv1(x, edges))
#     x = F.relu(self.conv2(x, edges))
#     x = F.relu(self.conv3(x, edges))
#     x = F.relu(self.conv4(x, edges))
#     x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#     x = self.lin1(x)
#     x = self.lin2(x)
#     x = self.pred(x)
#     # F.log_softmax(x, dim=1)

#     return x
#     # return F.log_softmax(x, dim=1)



# class GCN(torch.nn.Module):
#   def __init__(self, num_features, hidden_channels, num_classes):
#     super(GCN, self).__init__()
#     # torch.manual_seed(12345)
#     self.conv1 = GCNConv(num_features, hidden_channels*2)
#     self.conv2 = GCNConv(hidden_channels*2, hidden_channels*2)
#     self.conv3 = GCNConv(hidden_channels*2, hidden_channels)
#     self.conv4 = GCNConv(hidden_channels, 64)
#     self.lin1 = Linear(64, 32)
#     self.lin2 = Linear(32, 16)
#     self.pred = Linear(16, num_classes)

#   def forward(self, x, edges, batch):
#     x = self.conv1(x, edges)
#     x = F.relu(x)
#     x = self.conv2(x, edges)
#     x = F.relu(x)
#     x = self.conv3(x, edges)
#     x = F.relu(x)
#     x = self.conv4(x, edges)
#     x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#     x = self.lin1(x)
#     x = self.lin2(x)
#     pred = self.pred(x)

#     return pred


# class GCN(torch.nn.Module):
#   def __init__(self, num_features, hidden_channels, num_classes):
#     super(GCN, self).__init__()
#     # torch.manual_seed(12345)
#     self.conv1 = GCNConv(num_features, hidden_channels)
#     self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
#     self.conv3 = GCNConv(hidden_channels*2, hidden_channels)
#     # self.conv4 = GCNConv(hidden_channels*2, hidden_channels)
#     self.lin1 = Linear(hidden_channels, 32)
#     self.lin2 = Linear(32, 16)
#     self.pred = Linear(16, num_classes)

#   def forward(self, x, edges, batch):
#     x = self.conv1(x, edges)
#     x = F.relu(x)
#     x = self.conv2(x, edges)
#     x = F.relu(x)
#     x = self.conv3(x, edges)
#     # x = F.relu(x)
#     # x = self.conv4(x, edges)
#     x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#     x = self.lin1(x)
#     x = self.lin2(x)
#     x = self.pred(x)

#     return x


# class GCN(torch.nn.Module):
#   def __init__(self, num_features, num_classes):
#     super(GCN, self).__init__()
#     # torch.manual_seed(12345)
#     self.conv1 = GCNConv(num_features, num_features)
#     self.conv2 = GCNConv(num_features, 64)
#     self.conv3 = GCNConv(64, 32)
#     self.conv4 = GCNConv(32, 16)
#     self.fc1 = Linear(16, num_classes)

#   def forward(self, x, edges, batch):
#     #! Semantics-preserving Reinforcement Learning Attack Against Graph Neural Networks for Malware Detection：GCN
#     x = self.conv1(x, edges)
#     # x = F.relu(x)
#     x = self.conv2(x, edges)
#     # x = F.relu(x)
#     x = self.conv3(x, edges)
#     # x = F.relu(x)
#     x = torch.tanh(self.conv4(x, edges))
#     # x = self.conv4(x, edges)
#     # x = F.relu(x)
#     x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#     x = self.fc1(x)

#     return x