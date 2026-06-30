# Diamond Counter — SAM2 + Review UI

Counts diamonds in jewelry images using SAM2 automatic segmentation, with a
browser UI to merge/split/delete/reclassify masks for an exact final count.

## Setup on AWS g4dn.large (T4 GPU)

1. Launch a `g4dn.large` instance using the **AWS Deep Learning AMI (Ubuntu 22.04)**
   — this comes with NVIDIA drivers + CUDA preinstalled, saving you a lot of pain.
2. Open inbound TCP port `8000` in the instance's security group (source: your IP).
3. SSH in and copy this project folder to the instance.
4. Run:
   ```bash
   chmod +x setup_ec2.sh
   ./setup_ec2.sh
   ```
5. Start the app:
   ```bash
   source venv/bin/activate
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
6. Open `http://<your-ec2-public-ip>:8000` in your browser.

## How to use

1. **Upload** the necklace/jewelry image. SAM2 runs automatic mask generation
   (~10-30s on a T4) and shows every candidate diamond as a colored overlay.
2. **Review**:
   - Colors = shape guess (blue=round, pink=marquise, yellow=baguette, gray=unknown).
   - Click a mask to select it (white outline). Click again to deselect.
3. **Fix mistakes**:
   - **Merge**: select 2+ masks that are actually one diamond SAM2 over-split → click "Merge Selected".
   - **Split**: select 1 mask that's actually 2+ diamonds SAM2 merged together → click "Split Mode" →
     click points across the mask to draw a cut line → click "Finish Split".
   - **Delete**: select false positives (glare, prongs, background) → click "Delete Selected".
   - **Click-Add**: switch to "Click-Add Diamond" mode and click any diamond SAM2 missed entirely.
   - **Relabel**: select mask(s) → choose correct shape from the dropdown.
4. **Counts** update live in the left sidebar, broken down by shape.
5. **Export JSON** when done — gives you total count, count by shape, and
   per-diamond bbox/centroid/area for audit trail.

## Notes on accuracy

- SAM2's automatic pass is a *starting point*, not the final answer — pavé/channel
  settings with touching stones will need manual split/merge passes.
- `points_per_side=48` in `mask_generator` controls grid density for the automatic
  pass; increase further (e.g. 64) if very small diamonds are being missed
  entirely, at the cost of more compute and more false positives to clean up.
- Shape classification (`classify_shape` in `main.py`) is a rough geometric
  heuristic (aspect ratio + fill ratio). Use the manual relabel dropdown as the
  source of truth — don't trust the auto-label for final reporting.

## File structure

```
diamond-counter/
├── app/
│   └── main.py          # FastAPI backend + SAM2 integration
├── static/
│   └── index.html        # Review UI (vanilla JS, no build step)
├── setup_ec2.sh           # One-time EC2 environment setup
├── requirements.txt
├── uploads/                # Uploaded images (created at runtime)
└── outputs/                # Exported JSON results (created at runtime)
```
