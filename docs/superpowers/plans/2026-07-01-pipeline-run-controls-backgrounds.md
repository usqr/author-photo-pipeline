# Pipeline Run Controls & Selectable Backgrounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the compare page whole-run and per-image control over which pipeline steps run (upscale, canvas extend, B&W, background-tone match) with a color-match slider, let the user pick one of three predefined backgrounds (rainbow / leafs / cork) OR upload a custom background image, show only the first and last existing version per image, and remove the top reference preview row.

**Architecture:** The pipeline (`rainbow_convert.py`) gains two small JSON-backed config layers — global `settings.json` (step toggles, background choice, global color-match amount, optional custom-background filename) and per-image `image_options.json` (per-image upscale / canvas-extend toggles and color-match amount). Pure resolution helpers turn those into per-file decisions the pass workers consult. The LAB background-tone matching, drop shadow, and cork/leaf backgrounds come from the `feat/lab-color-match-drop-shadow` branch; the working Gemini upscale + canvas-extend come from `fix/nano-banana-upscale-extend`; both are cherry-picked onto a fresh branch off `main`. The static `compare.html` gains a global run-controls panel (including a custom-background upload) and per-row option controls, drops the reference row, and gates the previous/baseline thumbnails on file existence. `server.py` persists the two config files, accepts a base64 background upload, and exposes the config over new GET endpoints.

**Tech Stack:** Python 3.10+ (Pillow, OpenCV, NumPy, rembg/BiRefNet, google-genai, CorridorKey), stdlib `http.server`, vanilla ES5 HTML/JS (no build step), pytest for unit tests.

## Global Constraints

- **Branch:** all work happens on `feat/pipeline-run-controls-backgrounds` (already created off `main`).
- **No new runtime dependencies** beyond what `install.sh` already installs. pytest is a dev-only tool, invoked as `python3 -m pytest`.
- **Backgrounds: three predefined plus one custom slot.** The predefined keys/files (all already in the repo/history):
  - `"rainbow"` → `rainbow.png`
  - `"leafs"` → `leafs.jpeg`
  - `"cork"` → `wafle.jpg`
  - `"custom"` → the file named by `settings.custom_background` (uploaded via the compare page, saved as `custom_bg.<ext>`). Falls back to `rainbow.png` if the custom file is absent.
- **Upload safety:** the server accepts only `.png/.jpg/.jpeg/.webp`, sanitizes the filename to its basename (no path traversal), and stores it under `BASE_DIR`.
- **Color-match slider is 0–100**, integer. `amount=50` MUST reproduce the previously tuned LAB strength `(0.45, 0.10, 0.10)`.
- **Config files are optional at runtime**: missing/corrupt `settings.json` or `image_options.json` fall back to defaults without raising.
- **Default settings**: all four steps ON, `bg_match_amount = 50`, `background = "rainbow"`.
- **Keep ES5 JS style** already used in `compare.html` (`var`, `function`, string concatenation — no template literals, no arrow functions in existing patterns unless the file already uses them).
- **Conventional commits** (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `rainbow_convert.py` | Pipeline. Add config loaders + pure resolution helpers; gate Pass 1 (upscale), Pass 1.5 (canvas extend) per-file/global; make Pass 3 B&W a toggle; make Pass 4 background selectable with per-file color-match amount. | Modify |
| `settings.json` | Global run settings (persisted by server, read by pipeline). Created on first save; pipeline supplies defaults if absent. | Create (default committed) |
| `image_options.json` | Per-image overrides. Created on first save; `{}` default. | Create (default committed) |
| `server.py` | Persist `settings.json` + `image_options.json` from rerun payload; expose `/api/settings` and `/api/image_options`; accept explicit `targets` list. | Modify |
| `compare.html` | Remove reference row; global run-controls panel; per-row option controls; existence-gated first/last thumbnails; send settings + options + targets in rerun payloads. | Modify |
| `leafs.jpeg`, `wafle.jpg`, `rainbow.png` | The three predefined background images. | Recover / import / keep |
| `custom_bg.<ext>` | Optional user-uploaded background (written by the server at runtime; gitignored). | Runtime only |
| `tests/test_pipeline_config.py` | Unit tests for the pure config/resolution helpers. | Create |
| `tests/test_pipeline_color.py` | Unit tests for `bg_match_strength`, `match_histogram`, `compute_ref_stats`, `background_path`. | Create |
| `AGENTS.md`, `CLAUDE.md` | Document the new controls, backgrounds, and config files. | Modify |

### Shared interfaces (defined in Task 2 / Task 4, consumed by Tasks 3, 5, 7, 8)

Constants in `rainbow_convert.py`:

```python
SETTINGS_PATH = BASE_DIR / "settings.json"
IMAGE_OPTIONS_PATH = BASE_DIR / "image_options.json"
BG_PATHS = {
    "rainbow": BASE_DIR / "rainbow.png",
    "leafs":   BASE_DIR / "leafs.jpeg",
    "cork":    BASE_DIR / "wafle.jpg",
}
DEFAULT_SETTINGS = {
    "steps": {"upscale": True, "canvas_extend": True, "bw": True, "bg_match": True},
    "bg_match_amount": 50,
    "background": "rainbow",
    "custom_background": "",
}
```

Functions (exact signatures):

```python
def clamp_amount(v) -> int                                   # → int in [0,100], 50 on garbage
def load_settings() -> dict                                  # validated, merged over DEFAULT_SETTINGS
def load_image_options() -> dict                             # {} on missing/corrupt
def opt_bool(options: dict, fname: str, key: str, default: bool) -> bool
def effective_upscale(settings: dict, options: dict, fname: str) -> bool
def effective_extend(settings: dict, options: dict, fname: str) -> bool
def effective_bg_match_amount(settings: dict, options: dict, fname: str) -> int   # 0 when step off
def bg_match_strength(amount) -> tuple[float, float, float]  # (L,a,b); 50 → (0.45,0.10,0.10)
def background_path(settings: dict) -> Path                  # "custom" → sanitized custom_background; falls back to rainbow.png
```

Server HTTP contract:
- `GET /api/settings` → JSON of `settings.json` (or `{}` if absent).
- `GET /api/image_options` → JSON of `image_options.json` (or `{}`).
- `POST /api/rerun` body may now include `settings` (dict, may carry `background:"custom"` + `custom_background`), `image_options` (dict), `targets` (list of filenames). Server writes the two config files (when present) before launching, and passes `targets` as CLI args to `rainbow_convert.py`.
- `POST /api/upload_bg` body `{filename, data_base64}` → saves the decoded image as `custom_bg.<ext>` (extension whitelist + basename sanitized) and returns `{"saved": "<name>"}` (empty on rejection).

---

## Task 1: Branch base — cherry-pick both feature commits + recover background assets

**Files:**
- Modify: `rainbow_convert.py`, `compare.html`, `.gitignore` (via cherry-pick)
- Create: `wafle.jpg` (from cherry-pick), `leafs.jpeg` (recovered from stash blob)
- Keep: `rainbow.png`

**Interfaces:**
- Consumes: `main` HEAD (`819b945`), commit `420dda9` (upscale/extend restore), commit `c1a03bc` (LAB match + cork bg), blob `d7a7b76fac999964bf5a47c0290dcaf84bf83eea` (`leafs.jpeg`).
- Produces: a branch containing working Gemini upscale + canvas extend, the LAB helpers (`compute_ref_stats`, `match_histogram`, `make_drop_shadow`), and all three background image files on disk.

> Context: `420dda9` and `c1a03bc` both branch from `main` and both edit `rainbow_convert.py` in the header/imports and pass regions, so the second cherry-pick will conflict. Resolve by **union**: keep the upscale/extend Pass 1 + Pass 1.5 code from `420dda9` AND the LAB helpers / color Pass 3+4 / header constants from `c1a03bc`.

- [ ] **Step 1: Confirm branch and clean tree**

Run: `git branch --show-current && git status --porcelain`
Expected: `feat/pipeline-run-controls-backgrounds` and empty status.

- [ ] **Step 2: Cherry-pick the upscale/canvas-extend restore**

```bash
git cherry-pick 420dda9
```
Expected: applies cleanly (it is `main + 1`). If it reports "nothing to commit / empty" because those file states already match, run `git cherry-pick --skip` and continue — the goal is only that the upscale/extend code is present. Verify:
```bash
grep -n 'aspect_ratio="1:1"' rainbow_convert.py   # Pass 1 & 1.5 present
grep -n 'def upscale_long_edge' rainbow_convert.py
```
Expected: both found.

- [ ] **Step 3: Cherry-pick the LAB color-match / cork commit (expect conflict)**

```bash
git cherry-pick c1a03bc
```
Expected: conflict in `rainbow_convert.py` (header/imports/passes) and possibly `compare.html`. `wafle.jpg` adds cleanly.

- [ ] **Step 4: Resolve the `rainbow_convert.py` conflict by union**

Open `rainbow_convert.py`. For each conflict block, keep BOTH intents:
- **Imports:** keep `from PIL import Image, ImageEnhance, ImageOps, ImageFilter` (union of both — `ImageOps` from base for B&W autocontrast, `ImageFilter` from `c1a03bc` for drop shadow). Remove the `from skimage.metrics import structural_similarity as ssim` line (dropped by `c1a03bc`).
- **Header constants:** keep `Image.MAX_IMAGE_PIXELS = None`, keep `RAINBOW_BG_PATH`/`BW_REF_PATH`/`RAINBOW_REF_PATH` (still referenced by B&W path we are re-enabling), and keep the `c1a03bc` additions `BG_MATCH_STRENGTH`, `SHADOW_*`, `SHADOW_COLOR`. Keep `TARGET_LONG_EDGE` and `GEMINI_MODEL` from the upscale side.
- **`apply_rating_adjustments`:** keep the `c1a03bc` **color** version (operates on RGB).
- **New helpers:** keep `compute_ref_stats`, `match_histogram`, `make_drop_shadow` from `c1a03bc`.
- **Pass 1 (`worker_pass1`) and Pass 1.5 (`worker_pass15`):** keep the `420dda9` versions (Gemini upscale + canvas extend with `aspect_ratio="1:1"`).
- **Pass 3 (`worker_pass3`):** keep the `c1a03bc` color-adjust version for now (B&W becomes a toggle in Task 4).
- **Pass 4 (`worker_pass4`):** keep the `c1a03bc` version (LAB match + `BG_PATH` + drop shadow) for now.
- **`main()` SSIM block:** keep it removed (per `c1a03bc`).

Then:
```bash
git add rainbow_convert.py compare.html wafle.jpg .gitignore
git cherry-pick --continue
```

- [ ] **Step 5: Verify the module still imports**

Run: `python3 -c "import rainbow_convert; print('import OK')"`
Expected: `import OK` (no ImportError, no NameError). If it fails on a missing symbol, fix the union in Step 4.

- [ ] **Step 6: Recover the leaf background image**

```bash
git cat-file -p d7a7b76fac999964bf5a47c0290dcaf84bf83eea > leafs.jpeg
python3 -c "from PIL import Image; Image.MAX_IMAGE_PIXELS=None; im=Image.open('leafs.jpeg'); print(im.size, im.mode)"
```
Expected: `(2048, 2048) RGB`.

- [ ] **Step 7: Confirm all three backgrounds are on disk**

Run: `ls -la rainbow.png leafs.jpeg wafle.jpg`
Expected: all three exist.

- [ ] **Step 8: Ensure background images are tracked (not gitignored)**

Run: `git check-ignore rainbow.png leafs.jpeg wafle.jpg || echo "none ignored"`
Expected: `none ignored`. If any is ignored, add a negation to `.gitignore` (e.g. `!leafs.jpeg`).

- [ ] **Step 9: Commit the recovered asset**

```bash
git add leafs.jpeg .gitignore
git commit -m "chore: recover leafs.jpeg background and confirm rainbow/cork assets"
```

---

## Task 2: Config model — settings + image-options loaders and resolution helpers

**Files:**
- Modify: `rainbow_convert.py` (add constants + helper functions near the other module-level helpers, e.g. just after `get_rating`)
- Create: `tests/test_pipeline_config.py`
- Create: `settings.json` (default), `image_options.json` (default `{}`)

**Interfaces:**
- Consumes: `BASE_DIR`, `json`, `Path` (already imported).
- Produces: `SETTINGS_PATH`, `IMAGE_OPTIONS_PATH`, `BG_PATHS`, `DEFAULT_SETTINGS`, `clamp_amount`, `load_settings`, `load_image_options`, `opt_bool`, `effective_upscale`, `effective_extend`, `effective_bg_match_amount`, `background_path` (see Shared interfaces). `bg_match_strength` is added in Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_config.py`:

```python
import json
import importlib
import rainbow_convert as rc


def test_clamp_amount_bounds_and_garbage():
    assert rc.clamp_amount(-5) == 0
    assert rc.clamp_amount(150) == 100
    assert rc.clamp_amount(50) == 50
    assert rc.clamp_amount("37") == 37
    assert rc.clamp_amount(None) == 50      # garbage → default midpoint
    assert rc.clamp_amount("abc") == 50


def test_load_settings_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "SETTINGS_PATH", tmp_path / "nope.json")
    s = rc.load_settings()
    assert s == rc.DEFAULT_SETTINGS
    # returned dict must be independent of the module default
    s["steps"]["bw"] = False
    assert rc.DEFAULT_SETTINGS["steps"]["bw"] is True


def test_load_settings_merges_partial(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"steps": {"bw": False}, "background": "cork",
                             "bg_match_amount": 200}))
    monkeypatch.setattr(rc, "SETTINGS_PATH", p)
    s = rc.load_settings()
    assert s["steps"]["bw"] is False
    assert s["steps"]["upscale"] is True          # untouched key keeps default
    assert s["background"] == "cork"
    assert s["bg_match_amount"] == 100             # clamped


def test_load_settings_rejects_bad_background(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"background": "sparkles"}))
    monkeypatch.setattr(rc, "SETTINGS_PATH", p)
    assert rc.load_settings()["background"] == "rainbow"


def test_load_settings_corrupt_file(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    monkeypatch.setattr(rc, "SETTINGS_PATH", p)
    assert rc.load_settings() == rc.DEFAULT_SETTINGS


def test_load_image_options(tmp_path, monkeypatch):
    p = tmp_path / "opts.json"
    monkeypatch.setattr(rc, "IMAGE_OPTIONS_PATH", p)
    assert rc.load_image_options() == {}
    p.write_text(json.dumps({"a.jpg": {"upscale": False}}))
    assert rc.load_image_options() == {"a.jpg": {"upscale": False}}
    p.write_text("garbage")
    assert rc.load_image_options() == {}


def test_effective_upscale_and_extend():
    s = {"steps": {"upscale": True, "canvas_extend": False, "bw": True, "bg_match": True}}
    opts = {"x.jpg": {"upscale": False, "canvas_extend": True}}
    assert rc.effective_upscale(s, opts, "x.jpg") is False    # per-image override wins
    assert rc.effective_upscale(s, opts, "y.jpg") is True      # falls back to global
    assert rc.effective_extend(s, opts, "x.jpg") is True       # override enables it
    assert rc.effective_extend(s, opts, "y.jpg") is False      # global off


def test_effective_bg_match_amount():
    s = {"steps": {"bg_match": True}, "bg_match_amount": 50}
    assert rc.effective_bg_match_amount(s, {}, "z.jpg") == 50
    assert rc.effective_bg_match_amount(s, {"z.jpg": {"bg_match": 80}}, "z.jpg") == 80
    s_off = {"steps": {"bg_match": False}, "bg_match_amount": 50}
    assert rc.effective_bg_match_amount(s_off, {"z.jpg": {"bg_match": 80}}, "z.jpg") == 0


def test_background_path(monkeypatch, tmp_path):
    assert rc.background_path({"background": "cork"}) == rc.BG_PATHS["cork"]
    assert rc.background_path({"background": "leafs"}) == rc.BG_PATHS["leafs"]
    assert rc.background_path({"background": "???"}) == rc.BG_PATHS["rainbow"]
    assert rc.background_path({}) == rc.BG_PATHS["rainbow"]


def test_background_path_custom(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "BASE_DIR", tmp_path)
    # custom file present → returned
    (tmp_path / "custom_bg.png").write_bytes(b"x")
    assert rc.background_path(
        {"background": "custom", "custom_background": "custom_bg.png"}
    ) == tmp_path / "custom_bg.png"
    # custom selected but file missing → rainbow fallback
    assert rc.background_path(
        {"background": "custom", "custom_background": "gone.png"}
    ) == rc.BG_PATHS["rainbow"]
    # path traversal is stripped to basename (and then missing → fallback)
    assert rc.background_path(
        {"background": "custom", "custom_background": "../../etc/passwd"}
    ) == rc.BG_PATHS["rainbow"]


def test_load_settings_custom_background(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"background": "custom", "custom_background": "custom_bg.jpg"}))
    monkeypatch.setattr(rc, "SETTINGS_PATH", p)
    s = rc.load_settings()
    assert s["background"] == "custom"
    assert s["custom_background"] == "custom_bg.jpg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_pipeline_config.py -q`
Expected: FAIL — `AttributeError: module 'rainbow_convert' has no attribute 'clamp_amount'` (etc.).

- [ ] **Step 3: Add constants and helpers to `rainbow_convert.py`**

Insert these constants in the module header near the other `*_PATH` definitions (after `SERVICE_ACCOUNT_PATH`):

```python
SETTINGS_PATH = BASE_DIR / "settings.json"
IMAGE_OPTIONS_PATH = BASE_DIR / "image_options.json"

BG_PATHS = {
    "rainbow": BASE_DIR / "rainbow.png",
    "leafs":   BASE_DIR / "leafs.jpeg",
    "cork":    BASE_DIR / "wafle.jpg",
}

DEFAULT_SETTINGS = {
    "steps": {"upscale": True, "canvas_extend": True, "bw": True, "bg_match": True},
    "bg_match_amount": 50,
    "background": "rainbow",
    "custom_background": "",
}
```

Add these functions immediately after `get_rating` (around line 209):

```python
def clamp_amount(v):
    """Coerce a colour-match amount to an int in [0, 100]; 50 on garbage."""
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        return 50
    return max(0, min(100, v))


def load_settings():
    """Global run settings, validated and merged over DEFAULT_SETTINGS.
    Missing or corrupt settings.json → a fresh copy of the defaults."""
    s = {
        "steps": dict(DEFAULT_SETTINGS["steps"]),
        "bg_match_amount": DEFAULT_SETTINGS["bg_match_amount"],
        "background": DEFAULT_SETTINGS["background"],
        "custom_background": DEFAULT_SETTINGS["custom_background"],
    }
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
        except Exception:
            data = {}
        if isinstance(data.get("steps"), dict):
            for k in s["steps"]:
                if k in data["steps"]:
                    s["steps"][k] = bool(data["steps"][k])
        if "bg_match_amount" in data:
            s["bg_match_amount"] = clamp_amount(data["bg_match_amount"])
        if data.get("background") in BG_PATHS or data.get("background") == "custom":
            s["background"] = data["background"]
        if isinstance(data.get("custom_background"), str):
            s["custom_background"] = data["custom_background"]
    return s


def load_image_options():
    """Per-image overrides: {fname: {upscale, canvas_extend, bg_match}}. {} on error."""
    if IMAGE_OPTIONS_PATH.exists():
        try:
            data = json.loads(IMAGE_OPTIONS_PATH.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def opt_bool(options, fname, key, default):
    entry = options.get(fname)
    if isinstance(entry, dict) and key in entry:
        return bool(entry[key])
    return default


def effective_upscale(settings, options, fname):
    return opt_bool(options, fname, "upscale", settings["steps"]["upscale"])


def effective_extend(settings, options, fname):
    return opt_bool(options, fname, "canvas_extend", settings["steps"]["canvas_extend"])


def effective_bg_match_amount(settings, options, fname):
    """Resolved colour-match amount (0..100) for one file. 0 when the step is off."""
    if not settings["steps"].get("bg_match", True):
        return 0
    entry = options.get(fname)
    if isinstance(entry, dict) and "bg_match" in entry:
        return clamp_amount(entry["bg_match"])
    return clamp_amount(settings["bg_match_amount"])


def background_path(settings):
    if settings.get("background") == "custom":
        name = Path(settings.get("custom_background") or "").name  # strip any directory
        if name:
            p = BASE_DIR / name
            if p.exists():
                return p
        return BG_PATHS["rainbow"]
    return BG_PATHS.get(settings.get("background"), BG_PATHS["rainbow"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pipeline_config.py -q`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit default config files and helpers**

Create `settings.json`:
```json
{
  "steps": {"upscale": true, "canvas_extend": true, "bw": true, "bg_match": true},
  "bg_match_amount": 50,
  "background": "rainbow",
  "custom_background": ""
}
```
Create `image_options.json`:
```json
{}
```
Add a line to `.gitignore` so uploaded custom backgrounds aren't committed:
```
custom_bg.*
```
Then:
```bash
git add rainbow_convert.py tests/test_pipeline_config.py settings.json image_options.json .gitignore
git commit -m "feat: add run settings + per-image option config model with resolution helpers"
```

---

## Task 3: Gate Pass 1 (upscale) and Pass 1.5 (canvas extend) on the resolved decisions

**Files:**
- Modify: `rainbow_convert.py` — `run_pipeline` signature + `worker_pass1` + `worker_pass15`; `main()` to load settings/options and pass them through.

**Interfaces:**
- Consumes: `effective_upscale`, `effective_extend`, `load_settings`, `load_image_options` (Task 2).
- Produces: `run_pipeline(files, ratings, regen_from, settings, options)` — two new params threaded from `main()`.

> Behaviour: when a file's upscale is OFF, Pass 1 copies the original into `step1_upscaled/` instead of calling Gemini (downstream still needs a step1 file). When a file's canvas-extend is OFF, Pass 1.5 leaves that file's step1 image as-is. If NO file wants a pass, that pass is marked skipped and drains its queue.

- [ ] **Step 1: Write the failing test (any-file predicate helper)**

Add to `tests/test_pipeline_config.py`:

```python
def test_any_wants_helpers():
    s = {"steps": {"upscale": False, "canvas_extend": False, "bw": True, "bg_match": True}}
    files = [type("F", (), {"name": n})() for n in ("a.jpg", "b.jpg")]
    opts = {"b.jpg": {"upscale": True}}
    assert rc.any_wants_upscale(s, opts, files) is True     # b overrides on
    assert rc.any_wants_extend(s, {}, files) is False        # nobody wants extend
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_pipeline_config.py::test_any_wants_helpers -q`
Expected: FAIL — `has no attribute 'any_wants_upscale'`.

- [ ] **Step 3: Add the predicate helpers**

After `background_path` in `rainbow_convert.py`:

```python
def any_wants_upscale(settings, options, files):
    return any(effective_upscale(settings, options, f.name) for f in files)


def any_wants_extend(settings, options, files):
    return any(effective_extend(settings, options, f.name) for f in files)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_pipeline_config.py::test_any_wants_helpers -q`
Expected: PASS.

- [ ] **Step 5: Thread settings/options through `run_pipeline` and `main`**

Change the signature:
```python
def run_pipeline(files, ratings, regen_from, settings, options):
```

In `main()`, after `ratings = load_ratings()` add:
```python
    settings = load_settings()
    options = load_image_options()
    print(f"Run settings: steps={settings['steps']} bg={settings['background']} "
          f"match={settings['bg_match_amount']}")
```
And change the call site:
```python
    run_pipeline(files, ratings, regen_from, settings, options)
```

- [ ] **Step 6: Gate Pass 1 per file**

In `worker_pass1`, replace the per-file body of the Gemini loop (the `for i, f in enumerate(files, 1):` block that calls `gemini_enhance`) so that files whose upscale is off are copied verbatim. Also short-circuit the whole pass when nobody wants upscale. Insert at the top of the pass (right after the `regen_from > 1` skip block and BEFORE the `if not SERVICE_ACCOUNT_PATH.exists()` block):

```python
            if not any_wants_upscale(settings, options, files):
                _log("PASS 1: Upscale off for all files — copying originals")
                for i, f in enumerate(files, 1):
                    write_progress(1, "Copy (upscale off)", i, total, f.name)
                    save_img(Image.open(f).convert("RGBA"), STEP1_DIR / f.name)
                    write_progress_file_done(1, f.name)
                    q1.put(f)
                write_progress_pass_done(1)
                return
```

Then inside the Gemini `for i, f in enumerate(files, 1):` loop, immediately after `fname = f.name`, add the per-file bypass:

```python
                if not effective_upscale(settings, options, fname):
                    write_progress(1, "Copy (upscale off)", i, total, fname)
                    save_img(Image.open(f).convert("RGBA"), STEP1_DIR / fname)
                    write_progress_file_done(1, fname)
                    q1.put(f)
                    continue
```

- [ ] **Step 7: Gate Pass 1.5 per file**

In `worker_pass15`, after the `regen_from > 1.5` skip block, add a whole-pass short-circuit (place it BEFORE the `if not SERVICE_ACCOUNT_PATH.exists()` check):

```python
            if not any_wants_extend(settings, options, files):
                _mark_pass_skipped(1.5, "Canvas Extend")
                _log("PASS 1.5: Canvas extend off for all files — skipping")
                _drain(q125, q15)
                return
```

Then inside the `while True:` loop, immediately after `fname = f.name`, add the per-file bypass (keep the existing `s1_path` existence check after it):

```python
                if not effective_extend(settings, options, fname):
                    write_progress(1.5, "Canvas Extend (off)", i, total, fname, "skipped")
                    write_progress_file_done(1.5, fname, "skipped")
                    q15.put(f)
                    continue
```

- [ ] **Step 8: Smoke-test that the module still imports and a dry no-op run works**

Run:
```bash
python3 -c "import rainbow_convert; print('import OK')"
python3 -m pytest tests/test_pipeline_config.py -q
```
Expected: `import OK`, all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add rainbow_convert.py tests/test_pipeline_config.py
git commit -m "feat: gate upscale (Pass 1) and canvas extend (Pass 1.5) per-file and per-run"
```

---

## Task 4: B&W toggle (Pass 3) + selectable background & per-file colour match (Pass 4)

**Files:**
- Modify: `rainbow_convert.py` — add `bg_match_strength`; re-enable B&W as a toggle in `worker_pass3`; make `worker_pass4` background-selectable with per-file strength.
- Create: `tests/test_pipeline_color.py`

**Interfaces:**
- Consumes: `effective_bg_match_amount`, `background_path`, `compute_ref_stats`, `match_histogram`, `apply_rating_adjustments`, `make_drop_shadow` (from cherry-pick), `REF_CDF`/B&W refs.
- Produces: `bg_match_strength(amount)`; a Pass 3 that emits grayscale when `settings.steps.bw` else colour; a Pass 4 that uses `background_path(settings)` and per-file `bg_match_strength(effective_bg_match_amount(...))`.

- [ ] **Step 1: Write the failing colour tests**

Create `tests/test_pipeline_color.py`:

```python
import numpy as np
import rainbow_convert as rc


def test_bg_match_strength_calibration():
    assert rc.bg_match_strength(0) == (0.0, 0.0, 0.0)
    l, a, b = rc.bg_match_strength(50)
    assert round(l, 2) == 0.45 and round(a, 2) == 0.10 and round(b, 2) == 0.10
    l, a, b = rc.bg_match_strength(100)
    assert round(l, 2) == 0.90 and round(a, 2) == 0.20 and round(b, 2) == 0.20
    assert rc.bg_match_strength(-10) == (0.0, 0.0, 0.0)   # clamped


def test_match_histogram_zero_strength_is_noop():
    rng = np.arange(0, 27, dtype=np.uint8).reshape(3, 3, 3)
    mask = np.ones((3, 3), dtype=bool)
    ref = rc.compute_ref_stats(np.full((4, 4, 3), 200, np.uint8))
    out = rc.match_histogram(rng.copy(), mask, ref, (0.0, 0.0, 0.0))
    assert np.array_equal(out, rng)


def test_match_histogram_moves_toward_reference():
    src = np.full((8, 8, 3), 40, np.uint8)         # dark foreground
    mask = np.ones((8, 8), dtype=bool)
    ref = rc.compute_ref_stats(np.full((8, 8, 3), 210, np.uint8))  # bright bg
    out = rc.match_histogram(src.copy(), mask, ref, (0.9, 0.2, 0.2))
    assert out.mean() > src.mean()                  # lightened toward bright ref


def test_compute_ref_stats_shape():
    mean, std = rc.compute_ref_stats(np.full((5, 5, 3), 128, np.uint8))
    assert mean.shape == (3,) and std.shape == (3,)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_pipeline_color.py -q`
Expected: FAIL — `has no attribute 'bg_match_strength'`.

- [ ] **Step 3: Add `bg_match_strength`**

After `background_path` in `rainbow_convert.py`:

```python
def bg_match_strength(amount):
    """Map a 0..100 colour-match amount to per-channel LAB transfer strength.
    Calibrated so amount=50 reproduces the tuned default (0.45, 0.10, 0.10):
    lightness (L) matched strongly, colour (a, b) matched weakly."""
    a = clamp_amount(amount) / 100.0
    return (a * 0.9, a * 0.2, a * 0.2)
```

- [ ] **Step 4: Run colour tests to verify pass**

Run: `python3 -m pytest tests/test_pipeline_color.py -q`
Expected: PASS.

- [ ] **Step 5: Re-enable B&W as a toggle in Pass 3**

The cherry-picked `worker_pass3` currently does colour-only adjustments. Restore the B&W path behind `settings.steps["bw"]`. First, make the local B&W histogram reference lazy so the module imports even if `bw.png` is absent — replace the top-level `REF_CDF` precompute block (if it exists after the union) with a cached function:

```python
_REF_CDF = None

def _ref_cdf():
    """Cached CDF of the B&W reference (bw.png) foreground, for local B&W fallback."""
    global _REF_CDF
    if _REF_CDF is None:
        ref_rgba = np.array(Image.open(BW_REF_PATH).convert("RGBA"))
        ref_gray = np.array(Image.open(BW_REF_PATH).convert("L"))
        fg = ref_gray[ref_rgba[:, :, 3] > 128]
        hist, _ = np.histogram(fg, bins=256, range=(0, 256))
        cdf = np.cumsum(hist).astype(float)
        cdf /= cdf[-1]
        _REF_CDF = cdf
    return _REF_CDF
```

In `worker_pass3`, thread the B&W step. At the top of the worker (after the `regen_from > 3` skip block), set up the Gemini client only when B&W is on:

```python
            do_bw = settings["steps"].get("bw", True)
            client = None
            bw_prompt = (
                "Convert this portrait photo to high-contrast black and white, "
                "matching the style of the reference image provided. "
                "The result should have: bright whites on skin highlights, "
                "deep rich blacks in hair and dark areas, sharp detail, "
                "and a clean professional look. Keep the person on a transparent/white "
                "background. Keep all details identical — same pose, expression, features. "
                "Output only the B&W image."
            )
            if do_bw and SERVICE_ACCOUNT_PATH.exists():
                try:
                    client = get_gemini_client()
                    _log("P3: Gemini B&W conversion connected")
                except Exception as e:
                    _log(f"P3: Gemini connect failed ({e}); local B&W fallback")
```

Then, inside the per-file loop, replace the single `img_adj, has_adj, adj_vals = apply_rating_adjustments(img, a_np, ratings, fname)` line with a branch that produces a `base_rgb` PIL image first:

```python
                if do_bw:
                    gray = None
                    if client is not None:
                        result = gemini_enhance(client, img.convert("RGB"), prompt=bw_prompt)
                        if result is not None:
                            gray = result.convert("L")
                    if gray is None:
                        gray_np = np.array(img.convert("L"))
                        fg = a_np > 128
                        if fg.any():
                            src = gray_np[fg]
                            sh, _ = np.histogram(src, bins=256, range=(0, 256))
                            scdf = np.cumsum(sh).astype(float); scdf /= scdf[-1]
                            ref_cdf = _ref_cdf()
                            mapping = np.array(
                                [min(int(np.searchsorted(ref_cdf, scdf[j])), 255) for j in range(256)],
                                dtype=np.uint8)
                            gray_np = mapping[gray_np]
                        gray = ImageOps.autocontrast(Image.fromarray(gray_np), cutoff=0.5)
                    base_rgb = gray.convert("RGB")
                    kind = "bw"
                else:
                    base_rgb = img.convert("RGB")
                    kind = "color"

                img_adj, has_adj, adj_vals = apply_rating_adjustments(base_rgb, a_np, ratings, fname)
```

Update the log line and progress label in that loop from the colour-only wording to use `kind` (e.g. `_log(f"[P3 {i}/{total}] {fname}: {kind}{adj_str}")`).

- [ ] **Step 6: Make Pass 4 background-selectable with per-file colour match**

In `worker_pass4`, replace the fixed `bg_img = Image.open(BG_PATH)...` line with the selected background:

```python
            bg_img = Image.open(background_path(settings)).convert("RGBA")
```
(Keep the existing center-crop-to-square block and `ref_stats = compute_ref_stats(...)` that follow.)

Then inside the per-file loop, replace the fixed-strength match block (the one guarded by `if np.any(np.asarray(BG_MATCH_STRENGTH) > 0):`) with a per-file amount:

```python
                strength = bg_match_strength(effective_bg_match_amount(settings, options, fname))
                if any(s > 0 for s in strength):
                    r, g, b, a = img.split()
                    fg_mask = np.array(a) > 0
                    matched = match_histogram(
                        np.array(Image.merge("RGB", (r, g, b))),
                        fg_mask, ref_stats, strength,
                    )
                    img = Image.merge("RGBA", (*Image.fromarray(matched).split(), a))
```
(Leave the drop-shadow composite and `save_img(...)` below unchanged.)

- [ ] **Step 7: Verify import + full unit suite**

Run:
```bash
python3 -c "import rainbow_convert; print('import OK')"
python3 -m pytest tests/ -q
```
Expected: `import OK`; all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add rainbow_convert.py tests/test_pipeline_color.py
git commit -m "feat: B&W toggle (Pass 3) and selectable background + per-file colour match (Pass 4)"
```

---

## Task 5: Server — persist config, expose GET endpoints, accept explicit targets

**Files:**
- Modify: `server.py` — add `/api/settings` + `/api/image_options` GET; write both config files in the rerun POST; pass `targets` to the CLI.

**Interfaces:**
- Consumes: rerun payload keys `settings`, `image_options`, `targets` (from Task 7/8).
- Produces: `settings.json`, `image_options.json` written on rerun; new GET endpoints; `targets` forwarded as positional CLI args to `rainbow_convert.py` (which already supports file targets via `main()`).

- [ ] **Step 1: Add the two GET endpoints**

In `do_GET`, alongside the existing `/api/ratings` branch, add:

```python
        elif self.path == "/api/settings":
            self.send_json_file(BASE / "settings.json")
        elif self.path == "/api/image_options":
            self.send_json_file(BASE / "image_options.json")
```
(`send_json_file` already returns `{"done":true}` on missing files; that is acceptable — the client treats a missing/odd body as "use defaults". No change needed there.)

- [ ] **Step 2: Persist settings + image_options in the rerun POST**

In `do_POST`, inside the `/api/rerun` / `/api/regenerate` branch, after the existing `ratings_path.write_text(...)` handling and before "Clear progress", add:

```python
            # Persist global run settings and per-image options
            settings_data = data.get("settings")
            if isinstance(settings_data, dict):
                (BASE / "settings.json").write_text(jmod.dumps(settings_data, indent=2))
            options_data = data.get("image_options")
            if isinstance(options_data, dict):
                (BASE / "image_options.json").write_text(jmod.dumps(options_data, indent=2))
```

- [ ] **Step 3: Forward explicit targets to the pipeline**

In the same branch, after the `if data.get("regen_from"):` block that extends `cmd`, add:

```python
            targets = data.get("targets")
            if isinstance(targets, list) and not data.get("regenerate"):
                cmd.extend([str(t) for t in targets])
```
(Regenerate-all ignores targets by design; `main()` treats trailing non-flag args as file targets.)

- [ ] **Step 4: Add the custom-background upload endpoint**

Add a new branch in `do_POST` (e.g. right before the `elif self.path == "/api/stop":` branch):

```python
        elif self.path == "/api/upload_bg":
            import json as jmod, base64
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode()
            try:
                data = jmod.loads(body)
            except Exception:
                data = {}
            ext = Path(str(data.get("filename", ""))).suffix.lower()
            b64 = data.get("data_base64", "")
            saved = ""
            if ext in (".png", ".jpg", ".jpeg", ".webp") and b64:
                try:
                    raw = base64.b64decode(str(b64).split(",")[-1])
                    saved = "custom_bg" + ext
                    (BASE / saved).write_bytes(raw)
                except Exception:
                    saved = ""
            self.send_response(200 if saved else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(jmod.dumps({"saved": saved}).encode())
```
(`data_base64` may be a full data URL — `data:image/png;base64,XXXX` — so we split on `,` and decode the tail.)

- [ ] **Step 5: Manual verification — endpoints and persistence**

Start the server, then in another shell exercise the endpoints:
```bash
python3 server.py & sleep 1
curl -s localhost:8787/api/settings
curl -s -X POST localhost:8787/api/rerun -H 'Content-Type: application/json' \
  -d '{"regenerate":false,"settings":{"steps":{"upscale":true,"canvas_extend":false,"bw":false,"bg_match":true},"bg_match_amount":70,"background":"cork"},"image_options":{"anna.jpg":{"upscale":false}},"targets":[]}'
sleep 1
cat settings.json
cat image_options.json
pkill -f server.py
```
Expected: `settings.json` shows `bw:false`, `background:"cork"`, `bg_match_amount:70`; `image_options.json` shows `{"anna.jpg": {"upscale": false}}`. (This will kick off a pipeline run writing to `pipeline.log`; that is fine — you can `pkill -f rainbow_convert.py` to stop it.)

- [ ] **Step 6: Manual verification — upload endpoint**

With the server running, upload a tiny PNG as a custom background and confirm it lands on disk:
```bash
python3 server.py & sleep 1
B64=$(python3 -c "import base64,io; from PIL import Image; b=io.BytesIO(); Image.new('RGB',(8,8),(10,20,30)).save(b,'PNG'); print(base64.b64encode(b.getvalue()).decode())")
curl -s -X POST localhost:8787/api/upload_bg -H 'Content-Type: application/json' \
  -d "{\"filename\":\"my bg.png\",\"data_base64\":\"$B64\"}"
ls -la custom_bg.png
# reject a disallowed extension
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8787/api/upload_bg -H 'Content-Type: application/json' \
  -d '{"filename":"evil.svg","data_base64":"'"$B64"'"}'
pkill -f server.py; rm -f custom_bg.png
```
Expected: first call returns `{"saved":"custom_bg.png"}` and `custom_bg.png` exists; the `.svg` call returns HTTP `400`.

- [ ] **Step 7: Commit**

```bash
git add server.py
git commit -m "feat(server): persist config, expose GET endpoints, accept targets, upload custom bg"
```

---

## Task 6: Compare page — remove reference row, gate first/last thumbnails on existence

**Files:**
- Modify: `compare.html` — delete the reference row + `buildRefRow`; render previous/baseline cards only if their images load; relabel to "first"/"last".

**Interfaces:**
- Consumes: existing `NR`, `PR` globals, `/api/run_info`.
- Produces: an author grid with no top reference row and self-hiding first/last thumbnails.

- [ ] **Step 1: Remove the Reference row markup**

Delete these two lines from the `<body>`:
```html
<h2>Reference</h2>
<div class="ref-row" id="ref-row"></div>
```

- [ ] **Step 2: Remove `buildRefRow` and its call**

Delete the entire `function buildRefRow(){ ... }` definition. In `loadDynamic`, delete the `buildRefRow();` call (leave `init();` and `checkRunning();`).

- [ ] **Step 3: Gate the "prev" (last) section on existence and relabel**

In `init()`, the row HTML currently always renders a `prev` `<div class="rsec">`. Replace the prev section string so (a) it only renders when `PR` is set and (b) each `img-p` hides its card if it 404s. Change the prev section to be built conditionally:

```javascript
(PR?('<div class="rsec">'+
'<div class="rlbl-sm cp">last ('+PR+')</div>'+
'<div class="author-images">'+
'<div class="card"><img class="img-p" src="'+PR+'/step3_bw/'+en(f.stem)+'.png" onerror="this.closest(\'.card\').style.display=\'none\'"><div class="label cp">Adjust</div></div>'+
'<div class="card"><img class="img-p" src="'+PR+'/step4_rainbow/'+en(f.name)+'" onerror="this.closest(\'.card\').style.display=\'none\'"><div class="label cp">BG</div></div>'+
'</div>'+
'</div>'):'')+
```

- [ ] **Step 4: Gate the "baseline" (first) section and relabel**

Replace the baseline `<div class="rsec">` section similarly (baseline images may be absent for some files):

```javascript
'<div class="rsec">'+
'<div class="rlbl-sm cb">first (baseline)</div>'+
'<div class="author-images">'+
'<div class="card"><img class="img-b" src="baseline_bw/'+en(f.stem)+'.png" onerror="this.closest(\'.card\').style.display=\'none\'"><div class="label cb">Adjust</div></div>'+
'<div class="card"><img class="img-b" src="baseline_rainbow/'+en(f.name)+'" onerror="this.closest(\'.card\').style.display=\'none\'"><div class="label cb">BG</div></div>'+
'</div>'+
'</div>';
```

- [ ] **Step 5: Update the current ("new") section labels for consistency**

In the same row HTML, the current section's two output cards are labelled "B&W"/"Rainbow" (or "Color"/"Leafs" after cherry-pick). Relabel them to the neutral "Adjust" and "BG" so the labels hold whichever step toggles are active:
```javascript
'<div class="card"><img class="img-n" src="'+NR+'/step3_bw/'+en(f.stem)+'.png"><div class="label cn">Adjust</div></div>'+
'<div class="card"><img class="img-n" src="'+NR+'/step4_rainbow/'+en(f.name)+'"><div class="label cn">BG</div></div>'+
```

- [ ] **Step 6: Manual verification**

```bash
python3 server.py & sleep 1
```
Open `http://localhost:8787/compare.html`. Confirm: no "Reference" heading/row at the top; each author row shows the current outputs plus a "last (...)" section only when a previous run exists and a "first (baseline)" section; missing thumbnails leave no broken-image icon (their cards vanish). Then `pkill -f server.py`.

- [ ] **Step 7: Commit**

```bash
git add compare.html
git commit -m "feat(compare): drop reference row, show only existing first/last versions"
```

---

## Task 7: Compare page — global run-controls panel

**Files:**
- Modify: `compare.html` — add a controls panel (step checkboxes, colour-match slider, background radios); load `/api/settings`; include `settings` in every rerun payload; update `PASS_NAMES`.

**Interfaces:**
- Consumes: `GET /api/settings`.
- Produces: a global `SETTINGS` JS object; `getSettings()` returning the current panel state; `settings` included in `rerun`/`regenerate`/`regenFrom` payloads.

- [ ] **Step 1: Add the panel markup**

Immediately after the `<div class="controls">…</div>` block (before `<div class="filter-bar">`), add:

```html
<div class="runcfg" id="runcfg">
  <div class="runcfg-steps">
    <label><input type="checkbox" id="st-upscale" checked> Upscale</label>
    <label><input type="checkbox" id="st-canvas_extend" checked> Canvas extend</label>
    <label><input type="checkbox" id="st-bw" checked> Convert to B&amp;W</label>
    <label><input type="checkbox" id="st-bg_match" checked> Match background tone</label>
  </div>
  <div class="runcfg-match">
    <span>Colour match</span>
    <input type="range" id="bg-match-amount" min="0" max="100" step="5" value="50">
    <span id="bg-match-val">50</span>
  </div>
  <div class="runcfg-bg">
    <span>Background:</span>
    <label><input type="radio" name="bgsel" value="rainbow" checked> Rainbow</label>
    <label><input type="radio" name="bgsel" value="leafs"> Leafs</label>
    <label><input type="radio" name="bgsel" value="cork"> Cork</label>
    <label><input type="radio" name="bgsel" value="custom"> Custom</label>
    <input type="file" id="bg-custom-file" accept="image/png,image/jpeg,image/webp" style="font-size:11px;max-width:160px">
  </div>
</div>
```

- [ ] **Step 2: Add panel styling**

In the `<style>` block, add:

```css
.runcfg{display:flex;flex-wrap:wrap;gap:18px;justify-content:center;align-items:center;background:#16213e;border:1px solid #333;border-radius:8px;padding:10px 16px;margin:0 auto 15px;max-width:900px;font-size:12px}
.runcfg label{display:inline-flex;align-items:center;gap:4px;cursor:pointer}
.runcfg-match,.runcfg-bg,.runcfg-steps{display:flex;align-items:center;gap:8px}
.runcfg input[type="range"]{width:120px;accent-color:#4ecca3}
#bg-match-val{width:24px;text-align:center;color:#4ecca3;font-weight:bold}
```

- [ ] **Step 3: Add `SETTINGS`, `getSettings`, wiring, and slider readout**

In the `<script>`, near the other globals, add:
```javascript
var SETTINGS={steps:{upscale:true,canvas_extend:true,bw:true,bg_match:true},bg_match_amount:50,background:"rainbow",custom_background:""};
var CUSTOM_BG="";   // filename of the last uploaded custom background
```
Add helper functions:
```javascript
function getSettings(){
  var bg="rainbow";var rs=document.getElementsByName("bgsel");
  for(var i=0;i<rs.length;i++){if(rs[i].checked)bg=rs[i].value;}
  return {steps:{
      upscale:document.getElementById("st-upscale").checked,
      canvas_extend:document.getElementById("st-canvas_extend").checked,
      bw:document.getElementById("st-bw").checked,
      bg_match:document.getElementById("st-bg_match").checked
    },
    bg_match_amount:parseInt(document.getElementById("bg-match-amount").value,10),
    background:bg,
    custom_background:CUSTOM_BG};
}
function applySettingsToPanel(s){
  if(!s||!s.steps)return;
  document.getElementById("st-upscale").checked=!!s.steps.upscale;
  document.getElementById("st-canvas_extend").checked=!!s.steps.canvas_extend;
  document.getElementById("st-bw").checked=!!s.steps.bw;
  document.getElementById("st-bg_match").checked=!!s.steps.bg_match;
  var amt=(s.bg_match_amount!==undefined)?s.bg_match_amount:50;
  document.getElementById("bg-match-amount").value=amt;
  document.getElementById("bg-match-val").textContent=amt;
  CUSTOM_BG=s.custom_background||"";
  var rs=document.getElementsByName("bgsel");
  for(var i=0;i<rs.length;i++){rs[i].checked=(rs[i].value===(s.background||"rainbow"));}
}
function wireRunCfg(){
  var sl=document.getElementById("bg-match-amount");
  sl.addEventListener("input",function(){document.getElementById("bg-match-val").textContent=sl.value;});
  var cf=document.getElementById("bg-custom-file");
  cf.addEventListener("change",function(){
    var file=cf.files[0];if(!file)return;
    var rd=new FileReader();
    rd.onload=function(){
      fetch("/api/upload_bg",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({filename:file.name,data_base64:rd.result})})
      .then(function(r){return r.json();})
      .then(function(d){
        if(d&&d.saved){
          CUSTOM_BG=d.saved;
          var rs=document.getElementsByName("bgsel");
          for(var i=0;i<rs.length;i++){rs[i].checked=(rs[i].value==="custom");}
          alert("Custom background uploaded: "+d.saved);
        }else{alert("Upload failed — use a PNG/JPG/WEBP image.");}
      }).catch(function(e){alert("Upload error: "+e);});
    };
    rd.readAsDataURL(file);
  });
}
```

- [ ] **Step 4: Load settings during `loadDynamic`**

In `loadDynamic`, after the ratings fetch and before `init();`, add:
```javascript
  try{ SETTINGS=await fetch("/api/settings").then(function(r){return r.json();}); }catch(e){}
  wireRunCfg();
  applySettingsToPanel(SETTINGS);
```
(If the endpoint returned `{"done":true}` because the file is absent, `applySettingsToPanel` sees no `.steps` and leaves the HTML defaults — which already match the desired defaults.)

- [ ] **Step 5: Include `settings` in every rerun payload**

Update the three payload builders:
- `rerun()`: `var payload=JSON.stringify({ratings:ged(),settings:getSettings()});`
- `regenerate()`: `var payload=JSON.stringify({ratings:ged(),regenerate:true,settings:getSettings()});`
- `regenFrom()`: `var payload=JSON.stringify({ratings:ged(),regenerate:true,regen_from:p,settings:getSettings()});`

- [ ] **Step 6: Update `PASS_NAMES` labels**

Replace the `PASS_NAMES` object with neutral names that fit the toggles:
```javascript
var PASS_NAMES={"1":"Upscale/Copy","1.5":"Canvas Extend","2":"BG Remove","3":"Adjust/B&W","4":"Background"};
```

- [ ] **Step 7: Manual verification**

```bash
python3 server.py & sleep 1
```
Open the compare page. Toggle "Convert to B&W" off, set Background = Cork, drag the colour-match slider to 70 (readout updates). Then choose a local image with the **Custom** file picker → an alert confirms the upload and the Custom radio auto-selects. Click **Regenerate All** → accept. In a second shell:
```bash
cat settings.json
ls -la custom_bg.*
```
Expected: `settings.json` shows `"bw":false`, `"bg_match_amount":70`, `"background":"custom"`, `"custom_background":"custom_bg.<ext>"`; the `custom_bg.<ext>` file exists. Stop with `pkill -f rainbow_convert.py; pkill -f server.py` (you need not wait for the full run).

- [ ] **Step 8: Commit**

```bash
git add compare.html
git commit -m "feat(compare): global run-controls panel (steps, colour-match, background)"
```

---

## Task 8: Compare page — per-image option controls

**Files:**
- Modify: `compare.html` — per-row upscale/canvas-extend checkboxes + colour-match slider; load `/api/image_options`; build `image_options` (overrides only) and changed `targets` for rerun payloads.

**Interfaces:**
- Consumes: `GET /api/image_options`, `SETTINGS`/`getSettings` (Task 7), `PREV_RATINGS`, `ged` (existing).
- Produces: `IMG_OPTS` (loaded overrides) + `imgOverrides` state; `getImageOptions()` (overrides differing from global); `getChangedTargets()`; both included in rerun payloads.

> Per-image defaults follow the global panel. We store an override for a file only when its value differs from the current global setting, keeping `image_options.json` small.

- [ ] **Step 1: Add per-row controls markup in `init()`**

Inside the row template, after the `<div class="sp" id="sl-'+i+'"></div>` slider panel and before that `.rsec` closes, insert a per-image options block:

```javascript
'<div class="imgopt" id="io-'+i+'">'+
'<label><input type="checkbox" id="io-up-'+i+'" onchange="oio('+i+')"> upscale</label>'+
'<label><input type="checkbox" id="io-ce-'+i+'" onchange="oio('+i+')"> canvas</label>'+
'<span class="io-match"><input type="range" id="io-bm-'+i+'" min="0" max="100" step="5" oninput="oioBm('+i+',this.value)"><span id="io-bmv-'+i+'"></span></span>'+
'</div>'+
```

- [ ] **Step 2: Add per-row control styling**

In `<style>`:
```css
.imgopt{display:flex;gap:12px;align-items:center;font-size:10px;color:#aaa;margin-top:4px;flex-wrap:wrap}
.imgopt label{display:inline-flex;align-items:center;gap:3px;cursor:pointer}
.imgopt input[type="range"]{width:90px;accent-color:#4ecca3;vertical-align:middle}
.io-match{display:inline-flex;align-items:center;gap:4px}
#io-bmv,[id^="io-bmv-"]{width:20px;text-align:center;color:#4ecca3}
```

- [ ] **Step 3: Add state + handlers**

Near the globals:
```javascript
var IMG_OPTS={};        // last-saved overrides from server
var imgOverrides={};    // live per-file overrides {name:{upscale,canvas_extend,bg_match}}
```
Handlers:
```javascript
function oio(i){
  var n=FILES[i].name;var o=imgOverrides[n]||(imgOverrides[n]={});
  o.upscale=document.getElementById("io-up-"+i).checked;
  o.canvas_extend=document.getElementById("io-ce-"+i).checked;
}
function oioBm(i,v){
  var n=FILES[i].name;var o=imgOverrides[n]||(imgOverrides[n]={});
  o.bg_match=parseInt(v,10);
  document.getElementById("io-bmv-"+i).textContent=v;
}
```

- [ ] **Step 4: Initialise per-row controls from global + loaded overrides**

Add a function that seeds each row's controls (global default, overridden by `IMG_OPTS`):
```javascript
function seedImgOpts(){
  var g=getSettings();
  FILES.forEach(function(f,i){
    var ov=IMG_OPTS[f.name]||{};
    var up=(ov.upscale!==undefined)?ov.upscale:g.steps.upscale;
    var ce=(ov.canvas_extend!==undefined)?ov.canvas_extend:g.steps.canvas_extend;
    var bm=(ov.bg_match!==undefined)?ov.bg_match:g.bg_match_amount;
    document.getElementById("io-up-"+i).checked=!!up;
    document.getElementById("io-ce-"+i).checked=!!ce;
    document.getElementById("io-bm-"+i).value=bm;
    document.getElementById("io-bmv-"+i).textContent=bm;
    // seed live state from any loaded override so it persists on rerun
    if(Object.keys(ov).length)imgOverrides[f.name]=JSON.parse(JSON.stringify(ov));
  });
}
```

- [ ] **Step 5: Build overrides + changed targets for the payload**

```javascript
function getImageOptions(){
  var g=getSettings();var out={};
  FILES.forEach(function(f,i){
    var o={};
    var up=document.getElementById("io-up-"+i).checked;
    var ce=document.getElementById("io-ce-"+i).checked;
    var bm=parseInt(document.getElementById("io-bm-"+i).value,10);
    if(up!==g.steps.upscale)o.upscale=up;
    if(ce!==g.steps.canvas_extend)o.canvas_extend=ce;
    if(bm!==g.bg_match_amount)o.bg_match=bm;
    if(Object.keys(o).length)out[f.name]=o;
  });
  return out;
}
function getChangedTargets(){
  var opts=getImageOptions();var t=[];
  FILES.forEach(function(f){
    var rNow=JSON.stringify(ratings[f.name]||{});
    var rPrev=JSON.stringify(PREV_RATINGS[f.name]||{});
    var oNow=JSON.stringify(opts[f.name]||{});
    var oPrev=JSON.stringify(IMG_OPTS[f.name]||{});
    if(rNow!==rPrev||oNow!==oPrev)t.push(f.name);
  });
  return t;
}
```

- [ ] **Step 6: Load `/api/image_options` and seed after init**

In `loadDynamic`, after the settings fetch, add:
```javascript
  try{ IMG_OPTS=await fetch("/api/image_options").then(function(r){return r.json();}); if(typeof IMG_OPTS!=="object"||IMG_OPTS.done)IMG_OPTS={}; }catch(e){IMG_OPTS={};}
```
And after `init();` (which builds the rows), call `seedImgOpts();`.

- [ ] **Step 7: Send `image_options` + `targets` in payloads**

Update the payload builders:
- `rerun()`: `var payload=JSON.stringify({ratings:ged(),settings:getSettings(),image_options:getImageOptions(),targets:getChangedTargets()});`
- `regenerate()`: `var payload=JSON.stringify({ratings:ged(),regenerate:true,settings:getSettings(),image_options:getImageOptions()});`
- `regenFrom()`: `var payload=JSON.stringify({ratings:ged(),regenerate:true,regen_from:p,settings:getSettings(),image_options:getImageOptions()});`

- [ ] **Step 8: Manual verification**

```bash
python3 server.py & sleep 1
```
Open the compare page. For one author, uncheck "canvas" and set its per-image colour-match slider to a value different from the global. Click **Rerun Changed** → accept. In a second shell:
```bash
cat image_options.json
cat ratings.json
```
Expected: `image_options.json` contains only that author with `{"canvas_extend":false,"bg_match":<value>}` (files matching global are omitted). The pipeline log (`pipeline.log`) should show it processing only the changed file(s). Stop with `pkill -f rainbow_convert.py; pkill -f server.py`.

- [ ] **Step 9: Commit**

```bash
git add compare.html
git commit -m "feat(compare): per-image upscale/canvas toggles and colour-match slider"
```

---

## Task 9: Docs + end-to-end verification

**Files:**
- Modify: `AGENTS.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: updated documentation; a confirmed end-to-end run.

- [ ] **Step 1: Update `AGENTS.md`**

In the "Pipeline Steps" and "Compare Page" sections, document: (a) Pass 3 B&W is now a toggle (`settings.steps.bw`); Pass 4 composites the chosen background (`rainbow`/`leafs`/`cork`, or a user-uploaded `custom` image) with per-file LAB colour matching; (b) the new `settings.json` (including `background` + `custom_background`) and `image_options.json` config files and their schema; (c) the compare-page global run-controls panel (with the custom-background upload) and per-image controls; (d) the reference row was removed and only the first/last existing versions are shown. Add `settings.json`, `image_options.json`, `leafs.jpeg`, `wafle.jpg` to the Key Files table.

- [ ] **Step 2: Update `CLAUDE.md` quick reference**

Add a line noting the compare page now controls which steps run, the background choice, and colour-match amount, persisted to `settings.json` / `image_options.json`.

- [ ] **Step 3: Full unit suite**

Run: `python3 -m pytest tests/ -q`
Expected: all tests PASS.

- [ ] **Step 4: End-to-end smoke run on a single image**

Pick one small source file that exists in `webp/` (list it: `ls webp | head`). Then, with the panel defaults, target just that file:
```bash
python3 server.py & sleep 1
curl -s -X POST localhost:8787/api/rerun -H 'Content-Type: application/json' \
  -d '{"ratings":{},"settings":{"steps":{"upscale":false,"canvas_extend":false,"bw":false,"bg_match":true},"bg_match_amount":50,"background":"leafs"},"image_options":{},"targets":["<ONE_FILENAME>"]}'
```
Watch `pipeline.log` until it finishes (or `tail -f pipeline.log`). Expected: Pass 1 logs "Copy (upscale off)", Pass 1.5 skips, Pass 3 logs "color", Pass 4 uses the leafs background; `step4_rainbow/<file>` is produced over the foliage image. Confirm the output opens and shows the subject on leaves:
```bash
ls -la step4_rainbow/<ONE_FILENAME>*
pkill -f server.py
```

- [ ] **Step 5: Commit docs**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: document run controls, selectable backgrounds, and config files"
```

- [ ] **Step 6: Push the branch**

```bash
git push -u origin feat/pipeline-run-controls-backgrounds
```

---

## Self-Review

**Spec coverage:**
- "check if the image has any previous versions and display the first and last version only if they exist" → Task 6 (existence-gated first/baseline + last/previous sections; prev section omitted when no previous run).
- "whole-run checkboxes: canvas upsizing, image upscaling, turning bw, matching background tone + slider" → Task 7 (panel) + Tasks 3–4 (pipeline honouring them) + Task 5 (persist).
- "import bg-match from the feature branch where it was solved" → Task 1 (cherry-pick `c1a03bc`; LAB helpers reused in Task 4).
- "per-image checkboxes: upscaling, canvas extension, slider for bg colour match amount" → Task 8 (per-row controls) + Tasks 3–4 (per-file resolution) + Task 5 (persist/targets).
- "remove the preview row of the target images" → Task 6, Step 1–2 (top Reference row removed).
- "3 predefined backgrounds (rainbow, leafs, cork)" → Task 1 (assets recovered/imported) + Task 4 (`background_path`) + Task 7 (radios). Note: the compare page's old "Leafs" label pointed at the cork image; corrected here so `leafs` = `leafs.jpeg` (foliage) and `cork` = `wafle.jpg` (cork/waffle texture).
- "custom background, not only the 3 predefined ones" → Task 2 (`custom_background` in settings + `background_path` "custom" branch with basename sanitize/fallback) + Task 5 (`/api/upload_bg` with extension whitelist) + Task 7 (Custom radio + file picker + upload). Uploaded file saved as `custom_bg.<ext>` (gitignored).

**Placeholder scan:** No TBD/"handle edge cases"/"write tests for the above" — every code step shows the code; every test step shows the test.

**Type consistency:** `settings` is a dict with `steps` (dict of bools), `bg_match_amount` (int), `background` (str); `options` is `{fname: {upscale, canvas_extend, bg_match}}`. `bg_match_strength` returns a 3-tuple used by `match_histogram`. `run_pipeline(files, ratings, regen_from, settings, options)` matches its single call site in `main()`. Server payload keys (`settings`, `image_options`, `targets`) match what Tasks 7–8 send and Task 5 reads. Endpoint paths (`/api/settings`, `/api/image_options`) match between Task 5 and Tasks 7–8.

**Assumptions recorded:** "first/last versions" resolve to baseline (first) and `run_info.previous` (last), per the chosen option; per-image defaults inherit the global panel and only differing values are persisted; drop shadow remains always-on (not part of the requested toggles).
