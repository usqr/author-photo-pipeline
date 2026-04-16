# Author Photo Pipeline - Project Context

## Purpose
Converts author portrait photos into stylized B&W images on rainbow gradient backgrounds, matching a hand-crafted reference style (see `rainbow_Gokce.jpg`).

## Pipeline Steps
1. **AI Upscale (Pass 1)** — Gemini Flash Image API ("Nano Banana"). Enhances and upscales all portrait photos. Falls back to copying originals if no `service_account.json`.
2. **Green Background (Pass 1.25)** — BiRefNet portrait model via `rembg` detects person, then composites over standard chroma green (0,177,64). Prepares image for green screen keying in Pass 2.
3. **Canvas Extend (Pass 1.5)** — Gemini Flash Image API. Extends canvas ~15% on each side, continuing the person's body/hair/clothing seamlessly while keeping the green background solid.
4. **Green Screen Keying (Pass 2)** — CorridorKey neural network keyer (`github.com/nikopueringer/CorridorKey`). Uses Hiera backbone with CNN refinement for high-fidelity color unmixing — preserves hair detail, motion blur, and semi-transparent edges. Generates coarse alpha hint from chroma threshold, then produces clean straight foreground + linear alpha. Includes despill (green cast removal) and despeckle (matte cleanup).
5. **B&W Conversion (Pass 3)** — Gemini converts to high-contrast B&W matching reference style. Falls back to local histogram matching (CDF-matched to `bw.png`). Then per-image adjustments from `ratings.json`: shadow/highlight curves, brightness, contrast, sharpness.
6. **Rainbow Composite (Pass 4)** — Center-crop to square, scale `rainbow.png` gradient to match image size (no downscaling), alpha composite.

## Architecture
Pipeline runs **6 passes concurrently in a threaded pipeline** — each pass is a worker thread connected by queues:

```
files → [P1: Gemini Upscale] → q1 → [P1.25: Green BG] → q125 → [P1.5: Gemini Extend] → q15 → [P2: CorridorKey] → q2 → [P3: Gemini B&W] → q3 → [P4: Rainbow]
```

As soon as P1 finishes one image, P1.25 can start on it while P1 works on the next. This overlaps Gemini API I/O waits with BiRefNet/CorridorKey GPU work. Thread safety via `RLock` for progress and `Lock` for printing. Each worker has `try/finally` with sentinel propagation so downstream workers never hang on errors.

Progress reported to `progress.json` (top-level) and `progress_log.json` (per-pass/per-file detail with timing and status) for the compare page to poll.

## Key Files
| File | Description |
|------|-------------|
| `rainbow_convert.py` | Main pipeline — 6-pass pipelined processing (Gemini + BiRefNet + CorridorKey) |
| `server.py` | HTTP server with API endpoints (`/api/progress`, `/api/progress_log`, `/api/rerun`, `/api/stop`, `/api/run_info`, `/api/files`, `/api/ratings`) |
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
| `detail` | Too busy | Needs more detail | informational |

## Compare Page
- Served via `server.py` at `http://localhost:8787/compare.html`
- Must use HTTP server (not `file://`) because filenames contain spaces requiring URL encoding
- 3-way comparison: current run (green) vs previous run (orange ghost markers) vs baseline (red, no ratings)
- **Rerun button**: POSTs current slider values to `/api/rerun`, saves as `ratings.json`, triggers pipeline
- **Regen from Pass**: Re-run from a specific pass number, reusing earlier step outputs
- **End All button**: POSTs to `/api/stop`, kills pipeline and server
- **Progress bar**: Polls `/api/progress` and `/api/progress_log` every 2s, shows active passes (e.g. "P1+P2+P3")
- **Progress accordion**: Expandable detail per pass — per-file status badges (done/processing/skipped/fallback/error), elapsed time per file and per pass, mini progress bars
- Sliders initialize to previous run's weights; orange ghost markers show where they were
- localStorage auto-save for crash recovery

## Run Folders
- Each run archived to `YYMMDD_HHMM/` with step1-4 subfolders
- `baseline_bw/` and `baseline_rainbow/` contain no-ratings output for comparison
- Generated output folders (`step3_bw/`, `step4_rainbow/`, baselines, run archives) are tracked in git
- Intermediate folders (`step1_upscaled/`, `step2_nobg/`) at the top level are gitignored

## Environment
- **Python 3.10+** (tested on 3.14)
- **Key packages**: Pillow, rembg, google-genai, scikit-image, opencv-python-headless, torch, torchvision, timm
- **GCP**: requires `service_account.json` for Gemini API (project: `gemini-image-generation-492101`, model: `gemini-2.5-flash-image`)
- **CorridorKey**: cloned to `CorridorKey/` subdir, model (~300MB) auto-downloads from HuggingFace on first run
- **macOS**: uses `open` command and `osascript` notification
- **AI models**: BiRefNet (~1GB) downloaded to `~/.u2net/` on first run; CorridorKey (~300MB) to `CorridorKey/CorridorKeyModule/checkpoints/`
