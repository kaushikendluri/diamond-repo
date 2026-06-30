"""
Diamond Counter — FastAPI backend
Runs SAM2 automatic mask generation on an uploaded jewelry image,
serves a review UI for merge/split/delete/classify, and tracks live counts.

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import os
import uuid
import base64
import json
from io import BytesIO
from typing import List, Optional

import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.environ.get("SAM2_CHECKPOINT_DIR", "checkpoints")
SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory session store: session_id -> { image, masks, shape_labels }
SESSIONS = {}

app = FastAPI(title="Diamond Counter")

# ---------------------------------------------------------------------------
# Load SAM2 once at startup
# ---------------------------------------------------------------------------
print(f"Loading SAM2 on device={DEVICE} ...")
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=48,          # dense grid — small objects like diamonds need this
    pred_iou_thresh=0.86,
    stability_score_thresh=0.90,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=20,     # filter tiny noise masks (in pixels)
)

predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mask_to_polygon(mask: np.ndarray):
    """Convert a binary mask to a simplified polygon (list of [x,y]) for frontend rendering."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist()


def mask_bbox_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = float(xs.mean()), float(ys.mean())
    return [int(x0), int(y0), int(x1), int(y1)], [cx, cy]


def classify_shape(mask: np.ndarray) -> str:
    """Rough geometric heuristic to pre-label shape; user can override in UI."""
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return "unknown"
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    area = mask.sum()
    bbox_area = w * h
    fill_ratio = area / bbox_area if bbox_area > 0 else 0
    aspect = max(w, h) / max(1, min(w, h))

    if aspect > 2.8:
        return "baguette"
    if fill_ratio < 0.55 and aspect > 1.4:
        return "marquise"
    return "round"


def masks_to_session_format(masks: List[dict], image_shape):
    """Convert SAM2 mask generator output into our session's list-of-dict format."""
    result = []
    for m in masks:
        seg = m["segmentation"]
        bbox, centroid = mask_bbox_centroid(seg)
        if bbox is None:
            continue
        result.append({
            "id": str(uuid.uuid4())[:8],
            "polygon": mask_to_polygon(seg),
            "bbox": bbox,
            "centroid": centroid,
            "area": int(seg.sum()),
            "shape": classify_shape(seg),
            "deleted": False,
        })
    return result


def encode_mask_rle(mask: np.ndarray) -> dict:
    """Store mask compactly (RLE-ish via PNG bytes) so we can re-derive polygons after merge/split."""
    success, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
    return base64.b64encode(buf).decode("utf-8")


def decode_mask_png(b64: str, shape) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return (img > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:12]
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img_path = os.path.join(UPLOAD_DIR, f"{session_id}.png")
    img.save(img_path)

    img_np = np.array(img)

    print(f"[{session_id}] Running SAM2 automatic mask generation on {img_np.shape} ...")
    raw_masks = mask_generator.generate(img_np)
    print(f"[{session_id}] Generated {len(raw_masks)} raw masks.")

    # Store raw masks (as PNG-encoded binary) for later merge/split ops
    raw_mask_store = {}
    session_masks = []
    for m in raw_masks:
        seg = m["segmentation"]
        bbox, centroid = mask_bbox_centroid(seg)
        if bbox is None:
            continue
        mid = str(uuid.uuid4())[:8]
        raw_mask_store[mid] = encode_mask_rle(seg)
        session_masks.append({
            "id": mid,
            "polygon": mask_to_polygon(seg),
            "bbox": bbox,
            "centroid": centroid,
            "area": int(seg.sum()),
            "shape": classify_shape(seg),
            "deleted": False,
        })

    SESSIONS[session_id] = {
        "image_path": img_path,
        "image_shape": img_np.shape,
        "masks": session_masks,
        "mask_store": raw_mask_store,
    }

    return {
        "session_id": session_id,
        "image_url": f"/api/image/{session_id}",
        "width": img_np.shape[1],
        "height": img_np.shape[0],
        "masks": session_masks,
    }


@app.get("/api/image/{session_id}")
def get_image(session_id: str):
    from fastapi.responses import FileResponse
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return FileResponse(sess["image_path"])


class PointClick(BaseModel):
    session_id: str
    x: float
    y: float


@app.post("/api/click_segment")
def click_segment(req: PointClick):
    """Click-to-segment fallback: use SAM2 image predictor with a single point prompt.
    Useful for adding a diamond SAM2's automatic pass missed."""
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    img_np = np.array(Image.open(sess["image_path"]).convert("RGB"))
    predictor.set_image(img_np)
    point_coords = np.array([[req.x, req.y]])
    point_labels = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best = masks[int(np.argmax(scores))].astype(np.uint8)

    bbox, centroid = mask_bbox_centroid(best)
    if bbox is None:
        raise HTTPException(400, "No mask found at that point")

    mid = str(uuid.uuid4())[:8]
    sess["mask_store"][mid] = encode_mask_rle(best)
    new_mask = {
        "id": mid,
        "polygon": mask_to_polygon(best),
        "bbox": bbox,
        "centroid": centroid,
        "area": int(best.sum()),
        "shape": classify_shape(best),
        "deleted": False,
    }
    sess["masks"].append(new_mask)
    return new_mask


class MaskAction(BaseModel):
    session_id: str
    mask_ids: List[str]


@app.post("/api/merge")
def merge_masks(req: MaskAction):
    """Merge 2+ masks into one (for stones SAM2 over-split)."""
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if len(req.mask_ids) < 2:
        raise HTTPException(400, "Need at least 2 mask ids to merge")

    shape = sess["image_shape"][:2]
    combined = np.zeros(shape, dtype=np.uint8)
    for mid in req.mask_ids:
        if mid not in sess["mask_store"]:
            continue
        m = decode_mask_png(sess["mask_store"][mid], shape)
        combined = np.logical_or(combined, m).astype(np.uint8)

    # remove old masks
    sess["masks"] = [m for m in sess["masks"] if m["id"] not in req.mask_ids]
    for mid in req.mask_ids:
        sess["mask_store"].pop(mid, None)

    new_id = str(uuid.uuid4())[:8]
    sess["mask_store"][new_id] = encode_mask_rle(combined)
    bbox, centroid = mask_bbox_centroid(combined)
    new_mask = {
        "id": new_id,
        "polygon": mask_to_polygon(combined),
        "bbox": bbox,
        "centroid": centroid,
        "area": int(combined.sum()),
        "shape": classify_shape(combined),
        "deleted": False,
    }
    sess["masks"].append(new_mask)
    return new_mask


class SplitRequest(BaseModel):
    session_id: str
    mask_id: str
    split_line: List[List[float]]  # polyline [[x,y],[x,y],...] drawn by user across the mask


@app.post("/api/split")
def split_mask(req: SplitRequest):
    """Split one mask into two along a user-drawn line (for stones SAM2 merged together)."""
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    if req.mask_id not in sess["mask_store"]:
        raise HTTPException(404, "Mask not found")

    shape = sess["image_shape"][:2]
    mask = decode_mask_png(sess["mask_store"][req.mask_id], shape)

    # Draw the split line onto a blank canvas, dilate it slightly, subtract from mask,
    # then take connected components as the two new pieces.
    line_canvas = np.zeros(shape, dtype=np.uint8)
    pts = np.array(req.split_line, dtype=np.int32)
    cv2.polylines(line_canvas, [pts], isClosed=False, color=1, thickness=3)

    cut_mask = np.logical_and(mask, np.logical_not(line_canvas)).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(cut_mask)

    if num_labels < 3:  # background + at least 2 pieces expected
        raise HTTPException(400, "Split line did not separate the mask into two regions")

    # remove old mask
    sess["masks"] = [m for m in sess["masks"] if m["id"] != req.mask_id]
    sess["mask_store"].pop(req.mask_id, None)

    new_masks = []
    for label in range(1, num_labels):
        piece = (labels == label).astype(np.uint8)
        if piece.sum() < 15:  # discard slivers
            continue
        bbox, centroid = mask_bbox_centroid(piece)
        if bbox is None:
            continue
        new_id = str(uuid.uuid4())[:8]
        sess["mask_store"][new_id] = encode_mask_rle(piece)
        new_mask = {
            "id": new_id,
            "polygon": mask_to_polygon(piece),
            "bbox": bbox,
            "centroid": centroid,
            "area": int(piece.sum()),
            "shape": classify_shape(piece),
            "deleted": False,
        }
        sess["masks"].append(new_mask)
        new_masks.append(new_mask)

    return {"new_masks": new_masks}


@app.post("/api/delete")
def delete_masks(req: MaskAction):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    sess["masks"] = [m for m in sess["masks"] if m["id"] not in req.mask_ids]
    for mid in req.mask_ids:
        sess["mask_store"].pop(mid, None)
    return {"deleted": req.mask_ids}


class RelabelRequest(BaseModel):
    session_id: str
    mask_id: str
    shape: str


@app.post("/api/relabel")
def relabel_mask(req: RelabelRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    for m in sess["masks"]:
        if m["id"] == req.mask_id:
            m["shape"] = req.shape
            return m
    raise HTTPException(404, "Mask not found")


@app.get("/api/state/{session_id}")
def get_state(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    masks = sess["masks"]
    counts = {}
    for m in masks:
        counts[m["shape"]] = counts.get(m["shape"], 0) + 1
    return {
        "masks": masks,
        "total": len(masks),
        "counts_by_shape": counts,
    }


@app.get("/api/export/{session_id}")
def export_results(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    masks = sess["masks"]
    counts = {}
    for m in masks:
        counts[m["shape"]] = counts.get(m["shape"], 0) + 1

    out = {
        "session_id": session_id,
        "total_diamonds": len(masks),
        "counts_by_shape": counts,
        "diamonds": [
            {
                "id": m["id"],
                "shape": m["shape"],
                "bbox": m["bbox"],
                "centroid": m["centroid"],
                "area_px": m["area"],
            }
            for m in masks
        ],
    }
    out_path = os.path.join(OUTPUT_DIR, f"{session_id}_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


app.mount("/static", StaticFiles(directory="static"), name="static")
