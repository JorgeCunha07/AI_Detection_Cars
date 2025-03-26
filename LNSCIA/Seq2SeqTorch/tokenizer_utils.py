# tokenizer_utils.py
import re
import unicodedata
import numpy as np

class SimpleTokenizer:
    def __init__(self, oov_token=None, filters=None):
        """
        Implementação simples de um tokenizer para substituir o Keras Tokenizer.
        """
        self.oov_token = oov_token
        # Caso não seja fornecido, definimos um conjunto básico de caracteres a filtrar
        self.filters = filters if filters is not None else '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        self.word_counts = {}
        self.word_index = {}
        self.index_word = {}
        # Se definido o token OOV, reservamos o índice 1 para ele
        if self.oov_token is not None:
            self.word_index[self.oov_token] = 1
            self.index_word[1] = self.oov_token
            self.word_counts[self.oov_token] = 1

    def fit_on_texts(self, texts):
        for text in texts:
            tokens = text.split()
            for token in tokens:
                # Retira os filtros (caracteres especiais) das extremidades
                token = token.strip(self.filters)
                if token == '':
                    continue
                if token in self.word_counts:
                    self.word_counts[token] += 1
                else:
                    self.word_counts[token] = 1
        # Ordenar as palavras por frequência (decrescente)
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        # Se existe OOV, começamos em 2, senão em 1
        index = 1 if self.oov_token is None else 2
        for word, count in sorted_words:
            if word == self.oov_token:
                continue
            if word not in self.word_index:
                self.word_index[word] = index
                self.index_word[index] = word
                index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = text.split()
            seq = []
            for token in tokens:
                # Se a palavra existe, usamos o índice
                if token in self.word_index:
                    seq.append(self.word_index[token])
                # Caso contrário, se temos OOV, usamos esse índice
                elif self.oov_token is not None:
                    seq.append(self.word_index[self.oov_token])
                else:
                    seq.append(0)
            sequences.append(seq)
        return sequences


def pad_sequences(sequences, maxlen, padding='post'):
    """
    Função simples para aplicar padding a uma lista de sequências.
    :param sequences: lista de listas com índices
    :param maxlen: comprimento máximo de cada sequência
    :param padding: 'post' ou 'pre'
    :return: np.array com as sequências padronizadas
    """
    padded = []
    for seq in sequences:
        if len(seq) < maxlen:
            if padding == 'post':
                seq = seq + [0] * (maxlen - len(seq))
            else:
                seq = [0] * (maxlen - len(seq)) + seq
        else:
            seq = seq[:maxlen]
        padded.append(seq)
    return np.array(padded)
