# main.py
from fastapi import FastAPI
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from models import detect_labels_DataSet1, detect_labels_DataSet3
from models import detect_labels_DataSet2
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import Path, HTTPException
import os
from fastapi import Body
from routes.chatbot_routes import router as chatbot_router
from routes.description_routes import router as description_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router, prefix="/chatbot", tags=["ChatBot"])
app.include_router(description_router, prefix="/description", tags=["Description"])

class LabelInput(BaseModel):
    labels: List[str]


class SelectedModel(BaseModel):
    model: str


class Base64ImageInput(BaseModel):
    model: str
    image_base64: str


@app.get("/")
async def root():
    return {"message": "API para gerar descrições a partir de labels."}


@app.post("/image/findLabels/{dataset_id}")
async def find_labels_from_image_base64(
        dataset_id: int = Path(..., description="ID do dataset (1 ou 2)"),
        data: Base64ImageInput = Body(...)
):
    if dataset_id == 1:
        result, status = detect_labels_DataSet1(data.model, data.image_base64)
    elif dataset_id == 2:
        result, status = detect_labels_DataSet2(data.model, data.image_base64)
    elif dataset_id == 3:
        result, status = detect_labels_DataSet3(data.image_base64)
    else:
        return JSONResponse(
            content={"error": f"Dataset ID '{dataset_id}' não suportado."},
            status_code=400
        )

    return JSONResponse(content=result, status_code=status)


# Conjunto de datasets válidos
ALLOWED_DATASET_IDS = {1, 2, 3}


@app.get("/image/models/{dataset_id}")
def get_image_models(
        dataset_id: int = Path(..., description="ID do dataset (apenas 1 ou 2)")
):
    if dataset_id not in ALLOWED_DATASET_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset ID '{dataset_id}' não é suportado. Use um dos seguintes: {sorted(ALLOWED_DATASET_IDS)}"
        )

    models_path = os.path.join("modelsAvailable", str(dataset_id))

    if not os.path.exists(models_path):
        raise HTTPException(
            status_code=404,
            detail=f"Diretório para o dataset {dataset_id} não encontrado."
        )

    models = [
        f.split('.')[0]
        for f in os.listdir(models_path)
        if os.path.isfile(os.path.join(models_path, f))
    ]
    return models
