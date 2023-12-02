from torch.nn.parameter import Parameter as Param
from torch.nn.modules.module import Module as Mod
import torch.nn as nn
import torch
import math

class CustomAttention(Module):
  def __init__(self, input_size, output_size, negative_slope=0.2, num_heads=4, use_bias=True, use_residual=True):
    super(CustomAttention, self).__init__()
    self.num_heads = num_heads
    self.output_size = output_size
    self.weight = Param(torch.FloatTensor(input_size, num_heads * output_size))
    self.weight_u = Param(torch.FloatTensor(num_heads, output_size, 1))
    self.weight_v = Param(torch.FloatTensor(num_heads, output_size, 1))
    self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
    self.use_residual = use_residual
    if self.use_residual:
      self.project = nn.Linear(input_size, num_heads * output_size)
    else:
      self.project = None
    if use_bias:
      self.bias = Param(torch.FloatTensor(1, num_heads * output_size))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(-1))
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)
    self.weight.data.uniform_(-stdv, stdv)
    stdv = 1. / math.sqrt(self.weight_u.size(-1))
    self.weight_u.data.uniform_(-stdv, stdv)
    self.weight_v.data.uniform_(-stdv, stdv)

  def forward(self, inputs, adjacency_matrix, require_weight=False):
    support = torch.mm(inputs, self.weight)
    support = support.reshape(-1, self.num_heads, self.output_size).permute(dims=(1, 0, 2))
    f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
    f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
    logits = f_1 + f_2
    weight = self.leaky_relu(logits)
    masked_weight = torch.mul(weight, adjacency_matrix).to_sparse()
    attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
    support = torch.matmul(attn_weights, support)
    support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.output_size)
    if self.bias is not None:
      support = support + self.bias
    if self.use_residual:
      support = support + self.project(inputs)
    if require_weight:
      return support, attn_weights
    else:
      return support, None


class CustomPairNorm(nn.Module):
  def __init__(self, mode='PN', scale=1):
    assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
    super(CustomPairNorm, self).__init__()
    self.mode = mode
    self.scale = scale

  def forward(self, x):
    if self.mode == 'None':
      return x
    col_mean = x.mean(dim=0)
    if self.mode == 'PN':
      x = x - col_mean
      row_norm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
      x = self.scale * x / row_norm_mean
    if self.mode == 'PN-SI':
      x = x - col_mean
      row_norm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
      x = self.scale * x / row_norm_individual
    if self.mode == 'PN-SCS':
      row_norm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
      x = self.scale * x / row_norm_individual - col_mean
    return x


class CustomAttentionSemIndividual(Module):
  def __init__(self, input_size, hidden_size=128, activation=nn.Tanh()):
    super(CustomAttentionSemIndividual, self).__init__()
    self.project = nn.Sequential(nn.Linear(input_size, hidden_size),
                                  activation,
                                  nn.Linear(hidden_size, 1, bias=False))

  def forward(self, inputs, require_weight=False):
    w = self.project(inputs)
    beta = torch.softmax(w, dim=1)
    if require_weight:
      return (beta * inputs).sum(1), beta
    else:
      return (beta * inputs).sum(1), None


class CustomHeterogeneousGAT(nn.Module):
  def __init__(self, input_size=6, output_size=8, num_heads=8, hidden_dim=64, num_layers=1):
    super(CustomHeterogeneousGAT, self).__init__()
    self.encoding = nn.GRU(
      input_size=input_size,
      hidden_size=hidden_dim,
      num_layers=num_layers,
      batch_first=True,
      bidirectional=False,
      dropout=0.1
    )
    self.positive_attention = CustomAttention(
      input_size=hidden_dim,
      output_size=output_size,
      num_heads=num_heads
    )
    self.negative_attention = CustomAttention(
      input_size=hidden_dim,
      output_size=output_size,
      num_heads=num_heads
    )
    self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
    self.mlp_positive = nn.Linear(output_size*num_heads, hidden_dim)
    self.mlp_negative = nn.Linear(output_size*num_heads, hidden_dim)
    self.pair_norm = CustomPairNorm(mode='PN-SI')
    self.sem_attention = CustomAttentionSemIndividual(input_size=hidden_dim,
                                                      hidden_size=hidden_dim,
                                                      activation=nn.Tanh())
    self.predictor = nn.Sequential(
      nn.Linear(hidden_dim, 1),
      nn.Sigmoid()
    )

    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.02)

  def forward(self, inputs, pos_adjacency, neg_adjacency, require_weight=False):
    _, support = self.encoding(inputs)
    support = support.squeeze()
    pos_support, pos_attn_weights = self.positive_attention(support, pos_adjacency, require_weight)
    neg_support, neg_attn_weights = self.negative_attention(support, neg_adjacency, require_weight)
    support = self.mlp_self(support)
    pos_support = self.mlp_positive(pos_support)
    neg_support = self.mlp_negative(neg_support)
    all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
    all_embedding, sem_attn_weights = self.sem_attention(all_embedding, require_weight)
    all_embedding = self.pair_norm(all_embedding)
    if require_weight:
      return self.predictor(all_embedding), (pos_attn_weights, neg_attn_weights, sem_attn_weights)
    else:
      return self.predictor(all_embedding)
