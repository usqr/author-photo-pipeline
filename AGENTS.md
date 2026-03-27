# Author Photo Pipeline - Project Context

## Purpose
Author photo pipeline that converts portrait photos into stylized B&W images on rainbow gradient backgrounds.

## Pipeline Steps
1. **AI Upscale** - EDSR via `super_image` (also triggered when `detail > 0` in ratings, even for large images)
2. **Background Removal** - BiRefNet portrait model via `rembg`
3. **B&W Conversion** - Histogram-matched to `bw.png` reference, with per-image rating adjustments from `ratings.json`
4. **Rainbow Composite** - Overlay processed portrait on `rainbow.png` gradient background

## Architecture
Pipeline runs in **4 separate passes** to avoid OOM:
1. EDSR loaded/unloaded
2. BiRefNet loaded/unloaded
3. B&W conversion pass
4. Composite pass

## Key Files
| File | Description |
|------|-------------|
| `rainbow_convert.py` | Main pipeline, batched passes to avoid OOM |
| `ratings.json` | Per-image adjustments populated from `compare.html` |
| `compare.html` | Visual comparison tool with sliders (requires HTTP server, `file://` breaks URL-encoded filenames with spaces) |
| `bw.png` | B&W histogram reference |
| `rainbow.png` | Gradient background |
| `rainbow_Gokce.jpg` | Final output reference |

## Rating System
Ratings in `ratings.json` are per-image adjustments set via `compare.html` sliders.

**Scales:** all range from **-100 to +100**, where 0 = no change.

| Rating | Multiplier |
|--------|-----------|
| `brightness` | +/-0.8 |
| `contrast` | +/-0.8 |
| `sharpness` | +/-1.5 |
| `dark_areas` | +/-0.7-0.8 |
| `light_areas` | +/-0.7-0.8 |

Additional ratings: `lightness`, `pixelization`, `detail`.

**Special behavior:** `detail > 0` triggers EDSR AI upscaling even for images that are already large.

## Run Folders
- Each run is archived to `YYMMDD_HHMM` folders
- `baseline_bw/` and `baseline_rainbow/` contain output with no ratings applied

## Environment
- **Python 3.14**
- **Packages:** Pillow, rembg, super_image (includes torch), scikit-image, opencv-python-headless
- Compare page served via HTTP at `http://localhost:8787/compare.html`
