from collections import defaultdict
from typing import Sequence

import torch


class NgramInfo:
    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.info = defaultdict(list)
        self.offset = 0
        self.last_gram = None

    def reset(self):
        self.info = defaultdict(list)
        self.offset = 0
        self.last_gram = None

    def update_n_gram_statistics(self, text: Sequence):
        T = len(text)
        current_offset = self.offset

        if self.last_gram is not None and self.n > 1:
            text = self.last_gram[-(self.n - 1) :] + text
            current_offset -= self.n - 1

        for index in range(len(text) - self.n + 1):
            ngram = tuple(text[index : index + self.n])
            self.info[ngram].append(current_offset + index + len(ngram))

        self.last_gram = list(text[-self.n :])
        self.offset += T

    def get_n_gram_attentions(self, causal: bool = True, past_seen_tokens: int = 0) -> torch.Tensor:
        attention = torch.zeros((self.offset - past_seen_tokens, self.offset), dtype=torch.bool)
        for _, indices in self.info.items():
            if len(indices) > 1:
                indices = torch.tensor(indices)
                to_indices = indices[indices < (self.offset)]
                from_indices = indices[indices > past_seen_tokens] - 1 - past_seen_tokens
                attention[from_indices[:, None], to_indices] = True

        if causal:
            attention = torch.tril(attention, diagonal=past_seen_tokens)

        return attention


class AllGramInfo:
    def __init__(self, max_n: int = 3) -> None:
        self.max_n = max_n
        self.info = [NgramInfo(n) for n in range(1, max_n + 1)]

    def reset(self):
        self.info = [NgramInfo(n) for n in range(1, self.max_n + 1)]

    def update_all_gram_statistics(self, text: Sequence):
        # parallelize
        for info in self.info:
            info.update_n_gram_statistics(text)

    def get_all_gram_attentions(self, causal: bool = True, past_seen_tokens: int = 0) -> torch.Tensor:
        # parallelize
        attentions = [
            info.get_n_gram_attentions(causal=causal, past_seen_tokens=past_seen_tokens) for info in self.info
        ]
        return torch.stack(attentions)


class BatchAllGramInfo:
    def __init__(self, n: int = 3, batch_size: int = 1) -> None:
        self.batch_size = batch_size
        self.n = n
        self.info = [AllGramInfo(n) for _ in range(batch_size)]

    def reset(self):
        self.info = [AllGramInfo(self.n) for _ in range(self.batch_size)]

    def update_all_gram_statistics(self, text: Sequence):
        assert len(text) == self.batch_size
        # parallelize
        for i, t in enumerate(text):
            if isinstance(t, torch.Tensor):
                t = t.tolist()
            self.info[i].update_all_gram_statistics(t)

    def get_all_gram_attentions(self, causal: bool = True, past_seen_tokens: int = 0):
        # parallelize
        attentions = [
            info.get_all_gram_attentions(causal=causal, past_seen_tokens=past_seen_tokens) for info in self.info
        ]
        return torch.stack(attentions, dim=1)
