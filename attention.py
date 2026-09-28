import torch

class SelfAttention:
  def __init__(self, embedding, attention_dim, embedding_dim):
    self.query_weight = torch.empty(embedding_dim, attention_dim)
    self.key_weight = torch.empty(embedding_dim, attention_dim)
    self.value_weight = torch.empty(embedding_dim, attention_dim)
    self.embedding = embedding

    torch.nn.init.xavier_uniform_(self.query_weight)
    torch.nn.init.xavier_uniform_(self.key_weight)
    torch.nn.init.xavier_uniform_(self.value_weight)

    self.Q = torch.matmul(self.embedding, self.query_weight)
    self.K = torch.matmul(self.embedding, self.key_weight)
    self.V = torch.matmul(self.embedding, self.value_weight)