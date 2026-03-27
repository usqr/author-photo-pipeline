# Author Photo Pipeline - Project Context

## Purpose
Converts author portrait photos into stylized B&W images on rainbow gradient backgrounds, matching a hand-crafted reference style (see `rainbow_Gokce.jpg`).

## Pipeline Steps
1. **AI Upscale (Pass 1)** — EDSR via `super_image`. Small images (<500px) are always upscaled. Images with `detail > 0` in ratings are also upscaled regardless of size. Very small images (<250px) get denoised first via `cv2.fastNlMeansDenoisingColored`, then bilateral filtered after upscale to remove artifacts.
2. **Background Removal (Pass 2)** — BiRefNet portrait model via `rembg` with alpha matting for clean hair/edge separation. Parameters: foreground threshold 230, background threshold 20, erode size 6.
3. **B&W Conversion (Pass 3)** — Histogram-matched to `bw.png` reference using foreground-only CDF matching. Then per-image adjustments from `ratings.json`: shadow/highlight curves, brightness, contrast, sharpness. No CLAHE (causes over-darkening).
4. **Rainbow Composite (Pass 4)** — Center-crop to square, scale `rainbow.png` gradient to match image size (no downscaling), alpha composite.

## Architecture
Pipeline runs in **4 separate memory-isolated passes** to avoid OOM (BiRefNet ~973MB + EDSR can't coexist):
1. EDSR loaded → upscale all → EDSR freed
2. BiRefNet loaded → remove backgrounds → BiRefNet freed
3. B&W conversion (no heavy models, just PIL/numpy/OpenCV)
4. Rainbow composite (lightweight)

Progress is reported to `progress.json` for the compare page to poll.

## Key Files
| File | Description |
|------|-------------|
| `rainbow_convert.py` | Main pipeline — 4-pass batched processing |
| `server.py` | HTTP server with API endpoints (`/api/progress`, `/api/rerun`, `/api/stop`) |
| `ratings.json` | Per-image adjustments from compare.html sliders |
| `compare.html` | Visual comparison tool — 3-way (new/prev/baseline), sliders with ghost markers, progress bar, rerun/stop buttons |
| `bw.png` | B&W histogram reference (133x133, Gokce) |
| `rainbow.png` | Gradient background (280x280) |
| `rainbow_Gokce.jpg` | Final output reference (280x280, Gokce) |
| `install.sh` | Installs all Python dependencies |
| `run.sh` | Starts server, opens compare page, runs pipeline, archives to timestamped folder |

## Rating System
Ratings in `ratings.json` control per-image B&W adjustments. Set via compare.html sliders.

**Scales** (-100 to +100, 0 = no change):

| Rating | -100 means | +100 means | Multiplier range |
|--------|-----------|-----------|-----------------|
| `lightness` | Too bright (darken) | Too dark (brighten) | brightness ±0.8 |
| `contrast` | Too much (reduce) | Needs more (increase) | contrast ±0.8 |
| `dark_areas` | Too much black (lift shadows) | Needs deeper blacks (crush) | shadow curve ±0.7-0.8 |
| `light_areas` | Too blown out (pull down) | Needs brighter whites (push up) | highlight curve ±0.7 |
| `sharpness` | Oversharpened (soften) | Too soft (sharpen more) | sharpness ±1.5 |
| `pixelization` | Visible artifacts | Oversmoothed | informational (guides upscale decisions) |
| `detail` | Too busy | Needs more detail | **triggers EDSR upscale when > 0** |

## Compare Page
- Served via `server.py` at `http://localhost:8787/compare.html`
- Must use HTTP server (not `file://`) because filenames contain spaces requiring URL encoding
- 3-way comparison: current run (green) vs previous run (orange ghost markers) vs baseline (red, no ratings)
- **Rerun button**: POSTs current slider values to `/api/rerun`, saves as `ratings.json`, triggers pipeline
- **End All button**: POSTs to `/api/stop`, kills pipeline and server
- **Progress bar**: Polls `/api/progress` every 2s, shows pass/file/percentage
- Sliders initialize to previous run's weights; orange ghost markers show where they were
- localStorage auto-save for crash recovery

## Run Folders
- Each run archived to `YYMMDD_HHMM/` with step1-4 subfolders
- `baseline_bw/` and `baseline_rainbow/` contain no-ratings output for comparison
- Generated folders are gitignored (reproducible from source + ratings)

## Environment
- **Python 3.10+** (tested on 3.14)
- **Key packages**: Pillow, rembg, super-image (torch), scikit-image, opencv-python-headless
- **macOS**: uses `open` command and `osascript` notification
- **AI models** (~2GB): downloaded to `~/.u2net/` on first run
