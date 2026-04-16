# Author Photo Pipeline

Converts author portrait photos into stylized black & white images on rainbow gradient backgrounds, matching a reference style.

![Pipeline](rainbow_Gokce.jpg)

## Pipeline

| Pass | Step | Model | Description |
|------|------|-------|-------------|
| 1 | AI Upscale | Gemini Flash Image | Upscales and enhances all portrait photos via Gemini API. Falls back to copying originals if no service account. |
| 1.25 | Green Background | BiRefNet Portrait | Detects person via BiRefNet, composites over chroma green (0,177,64). Prepares for green screen keying. |
| 1.5 | Canvas Extend | Gemini Flash Image | Extends canvas ~15% on each side, continuing body/hair/clothing while keeping the green background solid. |
| 2 | Green Screen Keying | [CorridorKey](https://github.com/nikopueringer/CorridorKey) | Neural network keyer with Hiera backbone — high-fidelity color unmixing preserving hair detail and semi-transparent edges. Includes despill and despeckle. |
| 3 | B&W Conversion | Gemini Flash Image | Gemini converts to high-contrast B&W matching reference style, then per-image adjustments from `ratings.json`. Falls back to local histogram matching. |
| 4 | Rainbow Composite | — | B&W cutout composited on `rainbow.png` gradient (no downscaling). |

Passes run **concurrently in a pipeline** — each pass is a thread connected by queues. As soon as Pass 1 finishes an image, Pass 1.25 starts on it while Pass 1 works on the next image.

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
| Detail | Too busy | Needs more detail |

## Compare Page

The interactive compare page (`http://localhost:8787/compare.html`) provides:

- **3-way comparison** — Current run (green) vs previous run (orange) vs baseline with no ratings (red)
- **Ghost markers** — Orange markers on sliders show previous run's weight values
- **Hover to zoom** — 3.5x on current images, 4x on previous/baseline
- **Filters** — All / Has Issues / OK / Changed from previous
- **Progress bar** — Live pipeline progress showing active passes (e.g. "P1+P2+P3")
- **Progress accordion** — Expandable details per pass with per-file status (done/processing/skipped/error), timing, and mini progress bars
- **Rerun button** — Saves current weights and re-runs pipeline from the browser
- **Regen from Pass** — Re-run from a specific pass, reusing earlier step outputs
- **End All button** — Stops running pipeline and server
- **Export/Import JSON** — Save and load rating adjustments
- **Auto-save** — Slider values saved to localStorage

## Project Files

| File | Purpose |
|------|---------|
| `rainbow_convert.py` | Main pipeline (6-pass pipelined processing via Gemini + BiRefNet + CorridorKey) |
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
| `step2_nobg/` | Current run — green screen keyed (CorridorKey) |
| `step3_bw/` | Current run — B&W converted |
| `step4_rainbow/` | Current run — final rainbow composites |
| `baseline_bw/`, `baseline_rainbow/` | No-ratings output for comparison |
| `YYMMDD_HHMM/` | Archived previous runs |

## Requirements

- Python 3.10+
- macOS (uses `open` command for browser, `osascript` for notifications)
- GCP service account (`service_account.json`) for Gemini API access
- ~1GB disk for BiRefNet model (downloaded on first run to `~/.u2net/`)
- ~400MB disk for CorridorKey model (auto-downloads from HuggingFace on first run)
- PyTorch (for CorridorKey inference; uses MPS on Apple Silicon, CUDA on NVIDIA)
