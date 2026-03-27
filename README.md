# Author Photo Pipeline

Converts author portrait photos into stylized black & white images on rainbow gradient backgrounds, matching a reference style.

![Pipeline](rainbow_Gokce.jpg)

## Pipeline

| Pass | Step | Model | Description |
|------|------|-------|-------------|
| 1 | AI Upscale | EDSR | Upscales small/low-res images. `detail > 0` rating triggers upscale for any image. Includes denoising for very small sources (<250px). |
| 2 | Background Removal | BiRefNet Portrait | Removes backgrounds with alpha matting for clean hair/edge separation. |
| 3 | B&W Conversion | — | Histogram-matched to `bw.png` reference, then per-image brightness/contrast/sharpness/shadow/highlight adjustments from `ratings.json`. |
| 4 | Rainbow Composite | — | B&W cutout composited on `rainbow.png` gradient (no downscaling). |

Passes run sequentially with model loading/unloading between them to avoid OOM.

## Quick Start

```bash
# 1. Install dependencies
bash install.sh

# 2. Run pipeline + open compare page
bash run.sh

# 3. (Optional) Run only specific images
bash run.sh "image1.jpg" "image2.webp"
```

## Workflow

1. **Run** — `bash run.sh` starts the server, opens the compare page, and runs the pipeline
2. **Review** — Compare results in the browser at http://localhost:8787/compare.html
3. **Adjust** — Use sliders to rate images that need tweaking (orange ghost markers show previous weights)
4. **Re-run** — Click "Rerun with New Weights" in the browser, or export JSON → save as `ratings.json` → `bash run.sh`
5. **Iterate** — Repeat until satisfied. Each run is archived to a timestamped folder.

## Rating Scales

Each image can be rated on 7 scales (-100 to +100, 0 = no change):

| Scale | -100 | +100 |
|-------|------|------|
| Lightness | Too bright | Too dark |
| Contrast | Too much contrast | Needs more contrast |
| Dark Areas | Too much black | Needs deeper blacks |
| Light Areas | Too blown out | Needs brighter whites |
| Sharpness | Oversharpened | Too soft |
| Pixelization | Visible artifacts | Oversmoothed |
| Detail | Too busy | Needs more detail (triggers AI upscale) |

## Compare Page

The interactive compare page (`http://localhost:8787/compare.html`) provides:

- **3-way comparison** — Current run (green) vs previous run (orange) vs baseline with no ratings (red)
- **Ghost markers** — Orange markers on sliders show previous run's weight values
- **Hover to zoom** — 3.5x on current images, 4x on previous/baseline
- **Filters** — All / Has Issues / OK / Changed from previous
- **Progress bar** — Live pipeline progress with pass/file info
- **Rerun button** — Saves current weights and re-runs pipeline from the browser
- **End All button** — Stops running pipeline and server
- **Export/Import JSON** — Save and load rating adjustments
- **Auto-save** — Slider values saved to localStorage

## Project Files

| File | Purpose |
|------|---------|
| `rainbow_convert.py` | Main pipeline (4-pass batched processing) |
| `server.py` | HTTP server with API endpoints for progress/rerun/stop |
| `ratings.json` | Per-image adjustment weights |
| `compare.html` | Visual comparison & rating tool |
| `rainbow.png` | Rainbow gradient background |
| `bw.png` | B&W style reference |
| `rainbow_Gokce.jpg` | Final output reference |
| `install.sh` | Dependency installer |
| `run.sh` | Pipeline runner + server launcher |
| `AGENTS.md` | Full technical context for AI assistants |

## Folders

| Folder | Content |
|--------|---------|
| `webp/` | Source author photos |
| `step1_upscaled/` | Current run — upscaled images |
| `step2_nobg/` | Current run — background removed |
| `step3_bw/` | Current run — B&W converted |
| `step4_rainbow/` | Current run — final rainbow composites |
| `baseline_bw/`, `baseline_rainbow/` | No-ratings output for comparison |
| `YYMMDD_HHMM/` | Archived previous runs |

## Requirements

- Python 3.10+
- macOS (uses `open` command for browser, `osascript` for notifications)
- ~2GB disk for AI models (BiRefNet + EDSR, downloaded on first run to `~/.u2net/`)
