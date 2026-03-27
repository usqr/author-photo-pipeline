# Author Photo Pipeline

Converts author portrait photos into stylized black & white images on rainbow gradient backgrounds, matching a reference style.

## Pipeline Steps

1. **AI Upscale** — EDSR neural network upscales small/low-res images. Images with `detail > 0` rating also get upscaled regardless of size.
2. **Background Removal** — BiRefNet portrait model removes backgrounds with alpha matting for clean hair/edge separation.
3. **B&W Conversion** — Histogram-matched to `bw.png` reference, then per-image brightness/contrast/sharpness/shadow/highlight adjustments from `ratings.json`.
4. **Rainbow Composite** — Final B&W cutout composited on `rainbow.png` gradient at 280x280.

## Quick Start

```bash
# Install dependencies
bash install.sh

# Run pipeline (processes all images, opens compare page)
bash run.sh

# Run only specific images
bash run.sh "image1.jpg" "image2.webp"
```

## Workflow

1. Run the pipeline with `bash run.sh`
2. Review results in the compare page (http://localhost:8787/compare.html)
3. Adjust sliders for images that need tweaking
4. Export JSON and save as `ratings.json`
5. Re-run to apply adjustments

## Rating Scales

Each image can be rated on 7 scales (-100 to +100, 0 = no change needed):

| Scale | -100 | +100 |
|-------|------|------|
| Lightness | Too bright | Too dark |
| Contrast | Too much contrast | Needs more contrast |
| Dark Areas | Too much black | Needs deeper blacks |
| Light Areas | Too blown out | Needs brighter whites |
| Sharpness | Oversharpened | Too soft |
| Pixelization | Visible artifacts | Oversmoothed |
| Detail | Too busy | Needs more detail (triggers AI upscale) |

## Compare Page Features

- **3-way comparison**: Current run vs previous run vs baseline (no ratings)
- **Ghost markers**: Orange markers on sliders show previous run's weights
- **Hover to zoom**: 3.5x on current, 4x on previous/baseline images
- **Filter**: All / Issues / OK / Changed from previous
- **Export/Import JSON**: Save and load rating adjustments
- **Progress bar**: Shows pipeline progress during generation
- **Rerun button**: Save weights and re-run pipeline from the browser

## Files

| File | Purpose |
|------|---------|
| `rainbow_convert.py` | Main pipeline script (batched passes to avoid OOM) |
| `ratings.json` | Per-image adjustment weights |
| `compare.html` | Visual comparison & rating tool |
| `rainbow.png` | Rainbow gradient background |
| `bw.png` | B&W style reference |
| `rainbow_Gokce.jpg` | Final output reference |
| `install.sh` | Dependency installer |
| `run.sh` | Pipeline runner + server launcher |

## Requirements

- Python 3.10+
- macOS (uses `open` command for browser)
- ~2GB disk for AI models (downloaded on first run)
