# Author Photo Pipeline

See [AGENTS.md](AGENTS.md) for full project context.

## Quick reference
- `python3 install.sh` or `bash install.sh` to install dependencies
- `bash run.sh` to run the pipeline and open compare page
- Pipeline: rainbow_convert.py reads ratings.json for per-image adjustments
- Compare page served at http://localhost:8787/compare.html
- Compare page also controls which pipeline steps run (upscale, canvas extend, B&W, background colour-match), the background choice (rainbow/leafs/cork/custom upload), and the colour-match amount — globally and per-image; persisted to `settings.json` / `image_options.json`
