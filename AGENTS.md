# Author Photo Pipeline - Project Context

## Purpose
Converts author portrait photos into stylized B&W images on rainbow gradient backgrounds, matching a hand-crafted reference style (see `rainbow_Gokce.jpg`).

## Pipeline Steps
1. **AI Upscale (Pass 1)** — Gemini Flash Image API ("Nano Banana"). Enhances each portrait, squaring it via `aspect_ratio="1:1"` (the model no longer squares portraits by default — it preserves the input aspect ratio), then Lanczos-upscales to 2048 px on the long edge. NOTE: Nano Banana caps its own output at ~1 MP (~1024 px) regardless of the `image_size` config on this Vertex endpoint, so true 2K must be done locally (`upscale_long_edge`). Falls back to copying originals if no `service_account.json`.
2. **Green Background (Pass 1.25)** — BiRefNet portrait model via `rembg` detects person, then composites over standard chroma green (0,177,64). Prepares image for green screen keying in Pass 2.
3. **Canvas Extend (Pass 1.5)** — Gemini re-frames the portrait to a square (`aspect_ratio="1:1"`) and genuinely outpaints the cropped torso/shoulders/clothing so the body doesn't end in a hard cut, on a uniform chroma-green background. NOTE: the image is fed directly (no pre-padding) — padding green around it first just makes the model taper the body into a "bust" instead of generating new torso. Re-upscaled to 2048 px. Falls back to the upscaled input if Gemini fails.
4. **Green Screen Keying (Pass 2)** — CorridorKey neural network keyer (`github.com/nikopueringer/CorridorKey`). Uses Hiera backbone with CNN refinement for high-fidelity color unmixing — preserves hair detail, motion blur, and semi-transparent edges. Generates coarse alpha hint from chroma threshold, then produces clean straight foreground + linear alpha. Includes despill (green cast removal) and despeckle (matte cleanup).
5. **B&W Conversion (Pass 3)** — toggle: `settings.steps.bw` (default on). When on, Gemini converts to high-contrast B&W matching reference style, falling back to local histogram matching (CDF-matched to `bw.png`); logged as `kind="bw"`. When off, the image is kept in colour (`kind="color"`) — logged per-file as e.g. `[P3 i/N] file: color`. Either way, per-image adjustments from `ratings.json` (shadow/highlight curves, brightness, contrast, sharpness) are still applied.
6. **Background Composite (Pass 4)** — center-crop to square, then composite over the selected background: `rainbow` (`rainbow.png`), `leafs` (`leafs.jpeg`, foliage), `cork` (`wafle.jpg`, cork/waffle texture), or a user-uploaded `custom` image. Resolved by `background_path(settings)`; `custom` uses `settings.custom_background` (basename-sanitized) if that file exists under the project root, else falls back to `rainbow`. Before compositing, `bg_match_strength(effective_bg_match_amount(settings, options, fname))` maps a 0–100 "colour match" amount to per-channel LAB transfer strength (L weighted ~4.5x more than a/b) and blends the subject's LAB stats toward the background's via `match_histogram`; amount 0 (or `settings.steps.bg_match` off) skips matching entirely. A soft drop shadow (always on) is cast before the subject is alpha-composited on top.

## Architecture
Pipeline runs **6 passes concurrently in a threaded pipeline** — each pass is a worker thread connected by queues:

```
files → [P1: Gemini Upscale] → q1 → [P1.25: Green BG] → q125 → [P1.5: Gemini Extend] → q15 → [P2: CorridorKey] → q2 → [P3: Gemini B&W] → q3 → [P4: Background Composite]
```

As soon as P1 finishes one image, P1.25 can start on it while P1 works on the next. This overlaps Gemini API I/O waits with BiRefNet/CorridorKey GPU work. Thread safety via `RLock` for progress and `Lock` for printing. Each worker has `try/finally` with sentinel propagation so downstream workers never hang on errors.

Progress reported to `progress.json` (top-level) and `progress_log.json` (per-pass/per-file detail with timing and status) for the compare page to poll.

## Run Configuration
Two config files, both loaded fresh at the start of every run and tracked in git (so the last-used config is the checked-in default):

- **`settings.json`** — global run settings, read by `load_settings()`. Missing/corrupt file falls back to `DEFAULT_SETTINGS`. Schema:
  ```json
  {
    "steps": {"upscale": true, "canvas_extend": true, "bw": true, "bg_match": true},
    "bg_match_amount": 50,
    "background": "rainbow",
    "custom_background": ""
  }
  ```
  `steps.*` gate Pass 1 (upscale), Pass 1.5 (canvas_extend), Pass 3 (bw), and Pass 4 (bg_match). `bg_match_amount` is an int 0–100 (clamped by `clamp_amount()`). `background` is one of `rainbow` / `leafs` / `cork` / `custom`. `custom_background` names the uploaded file (e.g. `custom_bg.png`), only used when `background == "custom"`.
- **`image_options.json`** — sparse per-image overrides, read by `load_image_options()`. Shape: `{"<filename>": {"upscale": bool, "canvas_extend": bool, "bg_match": 0-100}}`. Any key omitted for a file falls back to the matching global `settings` value (`effective_upscale`, `effective_extend`, `effective_bg_match_amount` in `rainbow_convert.py`); compare.html only writes keys that actually differ from the global panel.

Both files are written by `server.py`'s `/api/rerun` (from the `settings` / `image_options` keys of the POST body) and read back via `GET /api/settings` / `GET /api/image_options` when compare.html reloads.

## Key Files
| File | Description |
|------|-------------|
| `rainbow_convert.py` | Main pipeline — 6-pass pipelined processing (Gemini + BiRefNet + CorridorKey) |
| `server.py` | HTTP server with API endpoints (`/api/progress`, `/api/progress_log`, `/api/rerun`, `/api/stop`, `/api/run_info`, `/api/files`, `/api/ratings`, `/api/settings`, `/api/image_options`, `/api/upload_bg`) |
| `ratings.json` | Per-image adjustments from compare.html sliders |
| `settings.json` | Global run settings (step toggles, background choice, colour-match amount) — see Run Configuration |
| `image_options.json` | Per-image setting overrides — see Run Configuration |
| `compare.html` | Visual comparison tool — global run-controls panel, per-image controls, first/last version display, progress bar, rerun/stop buttons |
| `bw.png` | B&W histogram reference (133x133, Gokce) |
| `rainbow.png` | Gradient background (280x280), `background: "rainbow"` |
| `leafs.jpeg` | Foliage background (2048x2048), `background: "leafs"` |
| `wafle.jpg` | Cork/waffle-texture background, `background: "cork"` |
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
- Per-author row shows the current run, plus "last" and "first (baseline)" sections **only if those versions exist** — there is no separate top-of-page reference row (removed; the old `.ref-row` markup is gone). "last" is only rendered `if(PR)` (previous run name from `run_info.json`'s `previous` key, empty on a first-ever run); "first (baseline)" is always emitted in markup but its `<img>` tags `onerror`-hide the card when `baseline_bw/`/`baseline_rainbow/` files don't exist for that image.
- **Global run-controls panel** (`#runcfg`): checkboxes for each step toggle (Upscale, Canvas extend, Convert to B&W, Match background tone), a Colour match slider (0–100), and a Background radio group (Rainbow / Leafs / Cork / Custom) with a file input to upload a custom background image (POSTs to `/api/upload_bg`; accepts png/jpg/jpeg/webp, saved server-side as `custom_bg.<ext>`). These populate the `settings` object sent with every rerun/regenerate.
- **Per-image controls**: each row has its own upscale/canvas-extend checkboxes and a colour-match slider; a value is only sent as an override (in `image_options`) when it differs from the current global panel value, otherwise the file inherits the global setting.
- **Rerun button**: POSTs current slider values plus `settings` + `image_options` + `targets` to `/api/rerun`, saves ratings as `ratings.json` and settings/options to `settings.json`/`image_options.json`, triggers pipeline
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
