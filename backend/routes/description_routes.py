from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Literal
from generator.seq2seq.inference import load_model, generate_sentence

router = APIRouter()

# Load model once when the server starts
model_seq2seq, input_vocab_seq2seq, output_vocab_seq2seq = load_model(
    model_path="generator/seq2seq/checkpoints/model_epoch50.pt",
    input_vocab_path="generator/seq2seq/checkpoints/input_vocab.pkl",
    output_vocab_path="generator/seq2seq/checkpoints/output_vocab.pkl"
)


# Request schema
class GenerateRequest(BaseModel):
    labels: List[str]
    mode: Literal["greedy", "topk", "beam"] = "greedy"
    topk: int = 5
    beam_width: int = 3
    temperature: float = 1.0


# Response schema (optional, for clarity)
class GenerateResponse(BaseModel):
    sentence: str


@router.post("/generate/seq2seq", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        sentence = generate_sentence(
            labels=request.labels,
            model=model_seq2seq,
            input_vocab=input_vocab_seq2seq,
            output_vocab=output_vocab_seq2seq,
            mode=request.mode,
            topk=request.topk,
            beam_width=request.beam_width,
            temperature=request.temperature
        )
        return {"sentence": sentence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
