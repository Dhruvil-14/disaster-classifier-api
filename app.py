"""
FastAPI service for the disaster type classifier.

Install:
    pip install -r requirements.txt

Run locally (set HF_MODEL_REPO_ID first):
    # Windows PowerShell:
    $env:HF_MODEL_REPO_ID="YOUR_USERNAME/disaster-classifier"
    uvicorn app:app --reload

    # Linux/macOS:
    export HF_MODEL_REPO_ID="YOUR_USERNAME/disaster-classifier"
    uvicorn app:app --reload

Then open:  http://127.0.0.1:8000/docs
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import os

import timm
from huggingface_hub import hf_hub_download
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import nn
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Keep CPU thread usage modest on small free instances.
torch.set_num_threads(1)

# ----------------------------- CONFIG ---------------------------------------
# Public Hugging Face model repository. Set this in Render's Environment.
# Example: YOUR_USERNAME/disaster-classifier
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "").strip()
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_model.pt").strip()
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "").strip() or None
REVIEW_THRESHOLD = 0.75  # below this confidence -> needs_review = True
# Stricter threshold for the weakest classes in testing (other_disaster F1 0.59, hurricane 0.72).
# Tune these on the dev set.
CLASS_THRESHOLDS = {"other_disaster": 0.70, "hurricane": 0.70}
MAX_FILE_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
# ----------------------------------------------------------------------------


class DisasterClassifier(nn.Module):
    """Must match the architecture used in training."""

    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# Loaded once at startup and reused for every request
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Render Free instances use CPU. Download the public checkpoint from
    # Hugging Face when the service starts; it is cached on the instance.
    if not HF_MODEL_REPO_ID:
        raise RuntimeError(
            "HF_MODEL_REPO_ID is missing. Set it to "
            "'YOUR_USERNAME/disaster-classifier' in Render Environment."
        )

    device = torch.device("cpu")
    print(
        f"Downloading {HF_MODEL_FILENAME} from "
        f"{HF_MODEL_REPO_ID} (revision={HF_MODEL_REVISION or 'default'})..."
    )
    ckpt_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=HF_MODEL_FILENAME,
        revision=HF_MODEL_REVISION,
        token=os.getenv("HF_TOKEN") or None,  # not required for a public repo
    )
    # Memory-conscious loading:
    # - mmap=True maps checkpoint tensors from disk instead of eagerly copying
    #   the entire 104 MB checkpoint into RAM.
    # - assign=True attaches checkpoint tensors to the model rather than
    #   copying them into a second set of parameter tensors.
    # This requires a recent PyTorch version and a checkpoint saved with the
    # standard zip-based torch.save format.
    try:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        use_assign = True
    except (TypeError, RuntimeError):
        # Compatibility fallback for older PyTorch/checkpoint formats.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        use_assign = False

    classes = ckpt["classes"]
    img_size = ckpt["img_size"]
    model_name = ckpt["model_name"]

    model = DisasterClassifier(model_name, len(classes))
    if use_assign:
        model.load_state_dict(ckpt["model"], assign=True)
    else:
        model.load_state_dict(ckpt["model"])

    # Release checkpoint dictionary references as soon as weights are attached.
    del ckpt
    import gc
    gc.collect()

    model = model.to(device)
    model.eval()

    STATE["device"] = device
    STATE["model"] = model
    STATE["classes"] = classes
    STATE["tf"] = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    print(
        f"Model loaded: {model_name} | classes: {classes} "
        f"| device: {device} | checkpoint: {ckpt_path}"
    )
    yield
    STATE.clear()


app = FastAPI(title="Disaster Image Classifier", version="1.0", lifespan=lifespan)

# Allow your React/Vue frontend to call this API (tighten the origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "classes": STATE.get("classes", [])}


# A plain `def` endpoint runs in a worker thread, so model inference
# does not block other requests.
@app.post("/predict")
def predict(
    file: Annotated[UploadFile, File(description="Disaster image (jpg/png/webp)")],
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            415, f"Unsupported file type '{file.content_type}'. Use jpg, png or webp."
        )

    data = file.file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_FILE_MB} MB).")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read this file as an image.") from None

    x = STATE["tf"](img).unsqueeze(0).to(STATE["device"])
    with torch.inference_mode():
        probs = torch.softmax(STATE["model"](x), dim=1)[0].cpu()

    classes = STATE["classes"]
    top_p, top_i = probs.topk(min(3, len(classes)))
    top3 = [
        {"label": classes[i], "confidence": round(p.item(), 4)}
        for p, i in zip(top_p, top_i, strict=True)
    ]

    label, confidence = top3[0]["label"], top3[0]["confidence"]
    threshold = CLASS_THRESHOLDS.get(label, REVIEW_THRESHOLD)

    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": confidence,
        "top3": top3,
        "all_probabilities": {
            c: round(p.item(), 4) for c, p in zip(classes, probs, strict=True)
        },
        "needs_review": confidence < threshold,
        "review_threshold_used": threshold,
    }
