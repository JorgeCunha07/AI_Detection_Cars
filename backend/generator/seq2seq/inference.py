import torch
import sys
import torch.nn.functional as F
from generator.seq2seq.model.encoder import Encoder
from generator.seq2seq.model.decoder import Decoder
from generator.seq2seq.model.seq2seq import Seq2Seq
from generator.seq2seq import vocab
from typing import List, Literal

sys.modules['vocab'] = vocab

from generator.seq2seq.vocab import Vocab, SOS, EOS, UNK

def load_model(
        model_path: str,
        input_vocab_path: str,
        output_vocab_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    input_vocab = Vocab.load(input_vocab_path)
    output_vocab = Vocab.load(output_vocab_path)

    encoder = Encoder(len(input_vocab), 256, 512)
    decoder = Decoder(len(output_vocab), 256, 512)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, input_vocab, output_vocab


def generate_sentence(
        labels: List[str],
        model: Seq2Seq,
        input_vocab: Vocab,
        output_vocab: Vocab,
        mode: Literal["greedy", "topk", "beam"] = "greedy",
        topk: int = 5,
        beam_width: int = 3,
        temperature: float = 1.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    model.eval()
    input_text = " ".join(labels)
    input_ids = input_vocab.encode(input_text)
    src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    encoder_outputs, hidden, cell = model.encoder(src_tensor)
    input_token = torch.tensor([output_vocab.word2idx[SOS]], dtype=torch.long).to(
        device
    )

    max_len = 20 + len(labels) * 5

    if mode == "beam":
        sequences = [[[], 0.0, hidden, cell, input_token]]
        for _ in range(max_len):
            all_candidates = []
            for seq, score, h, c, inp in sequences:
                output, h_new, c_new, _ = model.decoder(inp, h, c, encoder_outputs)
                log_probs = F.log_softmax(output, dim=1)
                topk_probs, topk_ids = log_probs.topk(beam_width)

                for i in range(beam_width):
                    word_id = topk_ids[0][i].item()
                    prob = topk_probs[0][i].item()
                    word = output_vocab.idx2word.get(word_id, UNK)

                    if word == EOS:
                        all_candidates.append((seq, score + prob, h_new, c_new, inp))
                    elif word != SOS:
                        next_input = torch.tensor([word_id], dtype=torch.long).to(
                            device
                        )
                        all_candidates.append(
                            (seq + [word], score + prob, h_new, c_new, next_input)
                        )

            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[
                        :beam_width
                        ]

        return " ".join(sequences[0][0])

    # greedy or topk
    output_sentence = []
    eos_count = 0

    for _ in range(max_len):
        output, hidden, cell, _ = model.decoder(
            input_token, hidden, cell, encoder_outputs
        )

        if mode == "topk":
            output = output / temperature
            probs = F.softmax(output, dim=1)
            topk_probs, topk_ids = probs.topk(topk)
            choice = torch.multinomial(topk_probs, 1).item()
            word_id = topk_ids[0][choice].item()
        else:
            word_id = output.argmax(1).item()

        word = output_vocab.idx2word.get(word_id, UNK)

        if word == EOS:
            eos_count += 1
            if eos_count >= 1:
                break
        elif word != SOS:
            output_sentence.append(word)

        input_token = torch.tensor([word_id], dtype=torch.long).to(device)

    return " ".join(output_sentence)
