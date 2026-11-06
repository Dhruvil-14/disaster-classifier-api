"""
FastAPI service for the disaster image classifier.

Render start command:
    uvicorn app:app --host 0.0.0.0 --port $PORT

Required Render environment variable:
    HF_MODEL_REPO_ID=Dhruvil-14/disaster-classifier

Optional:
    HF_MODEL_FILENAME=best_model.pt
    HF_MODEL_REVISION=main
    HF_TOKEN=...  # only needed for a private/gated Hugging Face repo
"""

import gc
import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Annotated

import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import nn
from torchvision import transforms

# ----------------------------- MEMORY SETTINGS -------------------------------
# Keep CPU thread usage low on small Render instances.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch may already have initialized its inter-op thread pool.
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------- CONFIG ----------------------------------------
HF_MODEL_REPO_ID = os.getenv(
    "HF_MODEL_REPO_ID", "Dhruvil-14/disaster-classifier"
).strip()
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_model.pt").strip()
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "").strip() or None

REVIEW_THRESHOLD = 0.75
CLASS_THRESHOLDS = {
    "other_disaster": 0.70,
    "hurricane": 0.70,
}

MAX_FILE_MB = 10
MAX_IMAGE_SIDE = 1024
ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

# Only one model inference at a time to avoid simultaneous RAM spikes.
PREDICT_LOCK = threading.Lock()

# Model and preprocessing pipeline are initialized once at startup.
STATE = {}


class DisasterClassifier(nn.Module):
    """Architecture must match the model used during training."""

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not HF_MODEL_REPO_ID:
        raise RuntimeError(
            "HF_MODEL_REPO_ID is missing. Set it to "
            "'Dhruvil-14/disaster-classifier' in Render Environment."
