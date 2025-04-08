import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from vocab import Vocab, SOS, EOS, PAD


class LabelToTextDataset(Dataset):
    def __init__(self, data, input_vocab=None, output_vocab=None):
        self.inputs = [" ".join(example["labels"]) for example in data]
        self.outputs = [f"{SOS} {example['description']} {EOS}" for example in data]

        self.input_vocab = input_vocab or Vocab(self.inputs)
        self.output_vocab = output_vocab or Vocab(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = torch.tensor(self.input_vocab.encode(self.inputs[idx]), dtype=torch.long)
        tgt = torch.tensor(self.output_vocab.encode(self.outputs[idx]), dtype=torch.long)
        return src, tgt

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.input_vocab.word2idx[PAD])
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.output_vocab.word2idx[PAD])
        return src_batch, tgt_batch


def load_data(path, val_split=0.1, seed=42):
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    train_data, val_data = train_test_split(raw_data, test_size=val_split, random_state=seed)
    train_dataset = LabelToTextDataset(train_data)
    val_dataset = LabelToTextDataset(val_data, train_dataset.input_vocab, train_dataset.output_vocab)

    return train_dataset, val_dataset
