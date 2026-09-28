import torch

class SelfAttention:
  def __init__(self, attention_dim, embedding_dim):
    self.query_weight = torch.empty(embedding_dim, attention_dim)
    self.key_weight = torch.empty(embedding_dim, attention_dim)
    self.value_weight = torch.empty(embedding_dim, attention_dim)

    torch.nn.init.xavier_uniform_(self.query_weight)
    torch.nn.init.xavier_uniform_(self.key_weight)
    torch.nn.init.xavier_uniform_(self.value_weight)

    self.query_weight.requires_grad_(True)
    self.key_weight.requires_grad_(True)
    self.value_weight.requires_grad_(True)

  def parameters(self):
    return [self.query_weight, self.key_weight, self.value_weight]

  def forward(self, embedding):
    Q = torch.matmul(embedding, self.query_weight)
    K = torch.matmul(embedding, self.key_weight)
    V = torch.matmul(embedding, self.value_weight)

    

