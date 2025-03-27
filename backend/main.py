from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class LabelInput(BaseModel):
    labels: List[str]
    
class SelectedModel(BaseModel):
    model: str


@app.get("/")
async def root():
    return {"message": "API para gerar descrições a partir de labels."}


@app.get("/image/models")
def get_image_models():
    return ["Modelo 1", "Modelo 2", "Modelo 1 + 2"]


@app.get("/description/models")
def get_description_models():
    return ["Custom", "Pre-built"]


@app.post("/image/findLabels")
def find_labels_from_image():
    return "Not implemented yet"


@app.post("/description/generate")
def generate_description(model: SelectedModel, data: LabelInput):
    return "Not implemented yet"
