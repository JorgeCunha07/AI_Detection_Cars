from collections import defaultdict

class SimpleTokenizer:
    def __init__(self, oov_token=None):
        self.oov_token = oov_token
        self.word_counts = defaultdict(int)
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts, min_freq=1):
        for text in texts:
            for word in text.strip().split():
                word = word.strip().lower()
                self.word_counts[word] += 1

        sorted_words = sorted(self.word_counts.items(), key=lambda x: -x[1])

        index = 1  # index 0 reservado para padding
        if self.oov_token:
            self.word_index[self.oov_token] = index
            self.index_word[index] = self.oov_token
            index += 1

        for word, count in sorted_words:
            if word == self.oov_token:
                continue
            if count < min_freq:
                continue
            self.word_index[word] = index
            self.index_word[index] = word
            index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = []
            for word in text.strip().split():
                word = word.strip().lower()
                idx = self.word_index.get(word, self.word_index.get(self.oov_token, 0))
                tokens.append(idx)
            sequences.append(tokens)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            words = [self.index_word.get(idx, self.oov_token) for idx in seq if idx != 0]
            texts.append(" ".join(words))
        return texts

def pad_sequences(sequences, maxlen, padding='post'):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if padding == 'post':
                padded.append(seq[:maxlen])
            else:
                padded.append(seq[-maxlen:])
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                padded.append(seq + [0]*pad_len)
            else:
                padded.append([0]*pad_len + seq)
    return padded
