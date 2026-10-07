import torch
import re
from collections import Counter, defaultdict
import heapq

class BPETokenizer:
  def __init__(self, vocab_size):
    self.vocab_size = vocab_size
    self.word_dict = {}
    self.char_dict = {}
    self.merges = {}
    self.word_freqs = {}
    self.vocab = set()

  def build_corpus(self, corpus):
    all_words = re.findall(r'\w+|[^\w\s]', corpus)
    self.word_dict = Counter(all_words)

    for word_id, (word, count) in enumerate(self.word_dict.items()):
      symbols = tuple(list(word) + ['</w>'])
      self.char_dict[word_id] = symbols
      self.word_freqs[word_id] = count

  def merge_tokens(self, first, second, merged, symbols):
    new_symbols = []
    i = 0

    while i<len(symbols):
      if i<len(symbols)-1 and symbols[i] == first and symbols[i+1] == second:
        new_symbols.append(merged)
        i += 2

      else:
        new_symbols.append(symbols[i])
        i += 1

    return tuple(new_symbols)

  def train(self, corpus):
    self.build_corpus(corpus)
    self.vocab = {char for symbols in self.char_dict.values() for char in symbols}

    pair_counts = Counter()
    inverted_index = defaultdict(set)

    for word_id, symbols in self.char_dict.items():
      freq = self.word_freqs[word_id]
      for i in range(len(symbols)-1):
        pair = (symbols[i], symbols[i+1])
        pair_counts[pair] += freq
        inverted_index[pair].add(word_id)

    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merge_count = self.vocab_size-len(self.vocab)+1

    while len(self.merges)<merge_count and heap: 
      neg_count, best_pair = heapq.heappop(heap)
      current_freq = -neg_count

      if best_pair in self.merges or pair_counts[best_pair] != current_freq or current_freq == 0:
        continue

      merged_token = "".join(best_pair)
      self.vocab.add(merged_token)
      self.merges[best_pair] = merged_token

      affected_word_ids = list(inverted_index[best_pair])
      pair_counts[best_pair] = 0
      del inverted_index[best_pair]

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