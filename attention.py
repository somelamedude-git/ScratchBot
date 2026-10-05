import torch
import re
from collections import Counter
import heapq

class BPETokenizer:
  def __init__(self, vocab_size):
    self.vocab_size = vocab_size
    self.word_dict = {}
    self.char_dict = {}
    self.merges = {}
    self.word_freqs = {}
    self.vocab = {}

  def build_corpus(self, corpus):
    all_words = re.findall(r'\w+|[^\w\s]', corpus)
    self.word_dict = Counter(all_words)

    for word_id, (word, count) in enumerate(self.word_dict.items()):
      symbols = tuple(list(word) + ['</w>'])
      self.char_dict[word_id] = symbols
      self.word_freqs[word_id] = count

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

    attention_score = torch.matmul(Q, torch.transpose(K, 0, 1))
    attention_score = attention_score/torch.sqrt(K.shape[-1])

    attention_weights = torch.softmax(attention_score, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output