import pickle
from collections import Counter

SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"


class Vocab:
    def __init__(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Tokens especiais
        self.word2idx = {
            PAD: 0,
            SOS: 1,
            EOS: 2,
            UNK: 3
        }

        # Adicionar palavras com frequência >= min_freq
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text: str):
        """Converte uma string de palavras para IDs"""
        return [self.word2idx.get(w, self.word2idx[UNK]) for w in text.split()]

    def decode(self, ids: list[int]):
        """Converte uma lista de IDs em string, ignorando tokens especiais"""
        return " ".join([
            self.idx2word.get(i, UNK)
            for i in ids
            if i not in [
                self.word2idx[SOS],
                self.word2idx[EOS],
                self.word2idx[PAD]
            ]
        ])

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
