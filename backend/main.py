# main.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os

from models import detect_labels

app = FastAPI()


class LabelInput(BaseModel):
    labels: List[str]

class SelectedModel(BaseModel):
    model: str


@app.get("/")
async def root():
    return {"message": "API para gerar descrições a partir de labels."}


@app.post("/image/findLabels")
async def find_labels_from_image(
    model: str = Query(..., description="Nome do modelo (sem extensão)"),
    file: UploadFile = File(...)
):
    file_bytes = await file.read()
    result, status = detect_labels(model, file_bytes)
    return JSONResponse(content=result, status_code=status)


@app.get("/image/models")
def get_image_models():
    models_path = "modelsAvailable"
    models = [f.split('.')[0] for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
    return models


@app.get("/description/models")
def get_description_models():
    return ["Custom", "Pre-built"]


@app.post("/description/generate")
def generate_description(model: SelectedModel, data: LabelInput):
    return "Not implemented yet"
