#!/usr/bin/env python3
"""
Multi-step author photo pipeline (v6) — memory-safe batched processing.
Runs each step as a separate pass to avoid OOM from loading all models at once.

  Pass 1:    AI upscale (Gemini Nano Banana) → step1_upscaled/
  Pass 1.25: Green background substitution (BiRefNet) → step1_upscaled/ (in-place)
  Pass 1.5:  Canvas extension (Gemini) → step1_upscaled/ (in-place)
  Pass 2:    Green screen keying (CorridorKey) → step2_nobg/
  Pass 3:    Color adjustments (per-image ratings) → step3_bw/
  Pass 4:    Background composite → step4_rainbow/

Reads ratings.json for per-image adjustments.
"""

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import json
import numpy as np
import sys
import time
import tempfile
import threading
from queue import Queue

Image.MAX_IMAGE_PIXELS = None  # background photos can be very high-res

BASE_DIR = Path(__file__).parent
WEBP_DIR = BASE_DIR / "webp"
BW_REF_PATH = BASE_DIR / "bw.png"
RATINGS_PATH = BASE_DIR / "ratings.json"
PREV_RATINGS_PATH = BASE_DIR / "prev_ratings.json"
SERVICE_ACCOUNT_PATH = BASE_DIR / "service_account.json"

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

GEMINI_PROJECT = "gemini-image-generation-492101"
GEMINI_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash-image"

STEP1_DIR = BASE_DIR / "step1_upscaled"
STEP2_DIR = BASE_DIR / "step2_nobg"
STEP3_DIR = BASE_DIR / "step3_bw"
STEP4_DIR = BASE_DIR / "step4_rainbow"
CORRIDORKEY_DIR = BASE_DIR / "CorridorKey"
CHROMA_GREEN = (0, 177, 64)  # Standard broadcast chroma green
TARGET_LONG_EDGE = 2048  # Lanczos upscale target; Nano Banana itself caps at ~1024 px

TARGET_SIZE = 280
AI_UPSCALE_THRESHOLD = 500

# Pass 4: blend the person's colour stats (mean/std) toward the background's in
# LAB space, as (L, a, b) strengths (0.0 = no change, 1.0 = full transfer). The
# per-run/per-file strength is resolved at composite time via bg_match_strength()
# + effective_bg_match_amount(); L (lightness) is weighted ~4.5x more than a/b so
# the subject integrates tonally without dragging in the background's colour cast.

# Pass 4: delicate drop shadow cast by the subject onto the background, so it
# reads as standing just in front of the wall. All sizes are fractions of the
# output side length. Set SHADOW_OPACITY = 0 to disable.
SHADOW_OPACITY = 0.4            # peak darkness of the shadow (0..1)
SHADOW_BLUR = 0.06             # gaussian blur radius (very diffuse, soft edge)
SHADOW_OFFSET_X = -0.03        # horizontal offset (negative = shadow to the left)
SHADOW_OFFSET_Y = 0.03        # vertical offset (positive = shadow downward)
SHADOW_COLOR = (25, 20, 15)    # dark warm-neutral, suits the tan background

PROGRESS_PATH = BASE_DIR / "progress.json"
PROGRESS_LOG_PATH = BASE_DIR / "progress_log.json"

# Thread safety for progress and printing
_progress_lock = threading.RLock()
_print_lock = threading.Lock()

# In-memory log of completed items per pass
_progress_log = {"_started": time.time(), "_elapsed": 0, "passes": {}}


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def _save_progress_log():
    """Must be called with _progress_lock held."""
    _progress_log["_elapsed"] = round(time.time() - _progress_log["_started"], 1)
    PROGRESS_LOG_PATH.write_text(json.dumps(_progress_log))


def write_progress(pass_num, pass_name, current, total, filename="", status="processing"):
    """Write progress to JSON file for compare page to poll."""
    with _progress_lock:
        data = {
            "pass": pass_num,
            "pass_name": pass_name,
            "current": current,
            "total": total,
            "filename": filename,
            "done": False,
        }
        PROGRESS_PATH.write_text(json.dumps(data))

        # Update detailed log
        key = str(pass_num)
        passes = _progress_log["passes"]
        if key not in passes:
            passes[key] = {
                "name": pass_name, "total": total, "started": time.time(),
                "elapsed": 0, "files": {},
            }
        p = passes[key]
        p["total"] = total
        p["elapsed"] = round(time.time() - p["started"], 1)
        if filename:
            if filename not in p["files"]:
                p["files"][filename] = {"status": status, "started": time.time(), "elapsed": 0}
            f = p["files"][filename]
            f["status"] = status
            f["elapsed"] = round(time.time() - f["started"], 1)
        _save_progress_log()


def write_progress_file_done(pass_num, filename, status="done"):
    """Mark a file as done/skipped/failed within a pass."""
    with _progress_lock:
        key = str(pass_num)
        passes = _progress_log["passes"]
        if key in passes and filename in passes[key]["files"]:
            f = passes[key]["files"][filename]
            f["status"] = status
            f["elapsed"] = round(time.time() - f["started"], 1)
            _save_progress_log()


def write_progress_pass_done(pass_num):
    """Mark a pass as complete."""
    with _progress_lock:
        key = str(pass_num)
        passes = _progress_log["passes"]
        if key in passes:
            passes[key]["elapsed"] = round(time.time() - passes[key]["started"], 1)
            passes[key]["done"] = True
            _save_progress_log()


def _mark_pass_skipped(pass_num, name):
    """Mark a pass as skipped in progress log."""
    with _progress_lock:
        _progress_log["passes"][str(pass_num)] = {
            "name": name + " (skipped)", "total": 0, "started": time.time(),
            "elapsed": 0, "files": {}, "done": True, "skipped": True,
        }
        _save_progress_log()


def write_progress_done():
    with _progress_lock:
        _progress_log["_elapsed"] = round(time.time() - _progress_log["_started"], 1)
        data = {"done": True, "pass": 4, "pass_name": "Complete", "current": 0, "total": 0, "filename": ""}
        PROGRESS_PATH.write_text(json.dumps(data))
        _save_progress_log()


def load_ratings():
    if RATINGS_PATH.exists():
        with open(RATINGS_PATH) as f:
            ratings = json.load(f)
        print(f"Loaded ratings for {len(ratings)} images")
        return ratings
    return {}


def load_prev_ratings():
    if PREV_RATINGS_PATH.exists():
        with open(PREV_RATINGS_PATH) as f:
            return json.load(f)
    return None


def save_prev_ratings(ratings):
    """Save current ratings as prev for next run's diff."""
    with open(PREV_RATINGS_PATH, "w") as f:
        json.dump(ratings, f, indent=2)


def get_changed_files(ratings, prev_ratings, all_files):
    """Return list of files whose ratings changed since last run.
    If no prev_ratings exist, return all files.
    """
    if prev_ratings is None:
        print("No previous ratings found — processing all files.")
        return all_files

    changed = []
    for f in all_files:
        name = f.name
        cur = ratings.get(name, {})
        prev = prev_ratings.get(name, {})
        if cur != prev:
            changed.append(f)

    if not changed:
        print("No rating changes detected — nothing to re-process.")
    else:
        print(f"{len(changed)} files with changed ratings (out of {len(all_files)} total):")
        for f in changed:
            print(f"  - {f.name}")

    return changed


def get_rating(ratings, filename, key, default=0):
    if filename in ratings and key in ratings[filename]:
        return ratings[filename][key]
    return default


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
    """Resolved colour-match amount (0..100) for one file. 0 when the step is off.

    The global bg_match step gates all per-file colour-match: when it is off,
    every per-file bg_match override is ignored (returns 0).
    """
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


def bg_match_strength(amount):
    """Map a 0..100 colour-match amount to per-channel LAB transfer strength.
    Calibrated so amount=50 reproduces the tuned default (0.45, 0.10, 0.10):
    lightness (L) matched strongly, colour (a, b) matched weakly."""
    a = clamp_amount(amount) / 100.0
    return (a * 0.9, a * 0.2, a * 0.2)


def any_wants_upscale(settings, options, files):
    return any(effective_upscale(settings, options, f.name) for f in files)


def any_wants_extend(settings, options, files):
    return any(effective_extend(settings, options, f.name) for f in files)


def save_img(img, path):
    suffix = path.suffix.lower()
    if suffix == ".png":
        img.save(path, "PNG")
    elif suffix == ".webp":
        if img.mode == "RGBA":
            img.save(path, "WEBP", quality=95)
        else:
            img.convert("RGB").save(path, "WEBP", quality=95)
    elif suffix in (".jpg", ".jpeg"):
        img.convert("RGB").save(path, "JPEG", quality=95)


def upscale_long_edge(img_pil, target=TARGET_LONG_EDGE):
    """Lanczos-upscale so the longest edge is `target` px (no-op if already
    larger). Nano Banana caps output at ~1 MP (~1024 px) regardless of the
    image_size config, so true 2K has to be done deterministically here."""
    w, h = img_pil.size
    long_edge = max(w, h)
    if long_edge >= target:
        return img_pil
    scale = target / long_edge
    return img_pil.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


# ── Pass 1: Upscale (Gemini Nano Banana) ─────────────────────────────────────

def gemini_enhance(client, img_pil, prompt=None, image_size=None, aspect_ratio=None):
    """Enhance a portrait image using Gemini image generation.
    Returns enhanced PIL Image or None on failure.

    image_size: one of "1K", "2K", "4K" — requested output resolution. The
        current Nano Banana default is 1K, aspect-preserving; pass "2K"/"4K"
        to actually upscale. (Older model revisions silently padded to 1024².)
    aspect_ratio: e.g. "1:1", "3:4" — forces output framing. Omit to preserve
        the input image's aspect ratio.
    """
    from google.genai import types as gtypes
    import time

    image_config = None
    if image_size or aspect_ratio:
        ic_kwargs = {}
        if image_size:
            ic_kwargs["image_size"] = image_size
        if aspect_ratio:
            ic_kwargs["aspect_ratio"] = aspect_ratio
        image_config = gtypes.ImageConfig(**ic_kwargs)

    if prompt is None:
        prompt = (
            "Upscale this portrait photo to higher resolution. "
            "CRITICAL: Do NOT change the person's face in ANY way. "
            "The face, eyes, nose, mouth, expression, skin tone, facial hair, "
            "wrinkles, and all facial features must remain PIXEL-PERFECT identical. "
            "Only improve resolution, reduce compression artifacts, and sharpen "
            "hair and clothing texture. The person must be completely recognizable "
            "as the exact same individual. Output only the enhanced photo."
        )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, img_pil],
                config=gtypes.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_config,
                ),
            )
            for part in response.parts:
                if part.inline_data is not None:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        tmp = tf.name
                    part.as_image().save(tmp)
                    result = Image.open(tmp).convert("RGBA")
                    Path(tmp).unlink(missing_ok=True)
                    return result
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Gemini error: {err[:150]}")
                return None
    return None


def get_gemini_client():
    """Create and return a Gemini client using service account credentials."""
    from google import genai
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_PATH),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        project=GEMINI_PROJECT,
        location=GEMINI_LOCATION,
        credentials=credentials,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def apply_rating_adjustments(rgb_pil, a_np, ratings, fname):
    """Apply per-image rating adjustments to a color image.
    Returns RGBA image with cleaned alpha."""
    r_light = get_rating(ratings, fname, "lightness", 0)
    r_contrast = get_rating(ratings, fname, "contrast", 0)
    r_dark = get_rating(ratings, fname, "dark_areas", 0)
    r_light_areas = get_rating(ratings, fname, "light_areas", 0)
    r_sharp = get_rating(ratings, fname, "sharpness", 0)

    # Gemini B&W can return a different size than the keyed alpha; align them
    # so the later putalpha/merge don't raise "images do not match".
    ah, aw = a_np.shape[:2]
    if rgb_pil.size != (aw, ah):
        rgb_pil = rgb_pil.resize((aw, ah), Image.LANCZOS)

    rgb_np = np.array(rgb_pil.convert("RGB"))

    # Curves for dark/light areas (applied per-channel)
    if r_dark != 0 or r_light_areas != 0:
        rgb_f = rgb_np.astype(np.float32)
        if r_dark < 0:
            lift = abs(r_dark) / 100.0 * 0.8
            m = rgb_f < 128
            rgb_f[m] = rgb_f[m] + lift * (128 - rgb_f[m])
        elif r_dark > 0:
            crush = r_dark / 100.0 * 0.7
            m = rgb_f < 128
            rgb_f[m] = rgb_f[m] * (1 - crush)
        if r_light_areas < 0:
            pull = abs(r_light_areas) / 100.0 * 0.7
            m = rgb_f > 128
            rgb_f[m] = rgb_f[m] - pull * (rgb_f[m] - 128)
        elif r_light_areas > 0:
            push = r_light_areas / 100.0 * 0.7
            m = rgb_f > 128
            rgb_f[m] = rgb_f[m] + push * (255 - rgb_f[m])
        rgb_np = np.clip(rgb_f, 0, 255).astype(np.uint8)

    rgb_pil = Image.fromarray(rgb_np)

    # Brightness
    brightness_factor = 1.0 + (r_light / 100.0) * 0.8
    if brightness_factor != 1.0:
        rgb_pil = ImageEnhance.Brightness(rgb_pil).enhance(max(0.2, brightness_factor))

    # Contrast
    contrast_factor = 1.0 + (r_contrast / 100.0) * 0.8
    if contrast_factor != 1.0:
        rgb_pil = ImageEnhance.Contrast(rgb_pil).enhance(max(0.2, contrast_factor))

    # Sharpness
    sharpness_factor = 1.0 + (r_sharp / 100.0) * 1.5
    if sharpness_factor != 1.0:
        rgb_pil = ImageEnhance.Sharpness(rgb_pil).enhance(max(0.0, sharpness_factor))

    has_adj = any(v != 0 for v in [r_light, r_contrast, r_dark, r_light_areas, r_sharp])

    # Clean alpha
    a_float = a_np.astype(np.float32)
    a_clean = np.clip((a_float - 20) * (255.0 / (235 - 20)), 0, 255).astype(np.uint8)

    img_out = rgb_pil.convert("RGBA")
    img_out.putalpha(Image.fromarray(a_clean))
    return img_out, has_adj, [r_light, r_contrast, r_dark, r_light_areas, r_sharp]


def compute_ref_stats(rgb_np):
    """Per-channel (mean, std) of the reference in CIELAB space (cv2)."""
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float64)
    flat = lab.reshape(-1, 3)
    return flat.mean(axis=0), flat.std(axis=0)


def match_histogram(rgb_np, fg_mask, ref_stats, strength):
    """Mean/std (Reinhard) colour transfer of the foreground toward a reference,
    performed in CIELAB space.

    Per channel, the foreground is recentred to the reference (leaf) mean and
    scaled by the reference/foreground std ratio. Working in LAB lets us match
    lightness (L) strongly so the subject sits in the scene tonally, while
    matching colour (a, b) weakly so little of the background's green is dragged
    in. Only foreground pixels (fg_mask) are modified. `strength` is a per-channel
    (L, a, b) blend between the original (0.0) and fully transferred (1.0)
    result; a scalar applies the same strength to all channels.
    """
    if not fg_mask.any():
        return rgb_np
    if np.isscalar(strength):
        strength = (strength, strength, strength)
    if not any(s > 0 for s in strength):
        return rgb_np
    ref_mean, ref_std = ref_stats
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float64)
    for c in range(3):
        s = strength[c]
        if s <= 0:
            continue
        src_fg = lab[:, :, c][fg_mask]
        src_mean = src_fg.mean()
        src_std = src_fg.std()
        scale = (ref_std[c] / src_std) if src_std > 1e-6 else 1.0
        transferred = (src_fg - src_mean) * scale + ref_mean[c]
        lab[:, :, c][fg_mask] = (1.0 - s) * src_fg + s * transferred
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


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


def make_drop_shadow(alpha, size):
    """Build a soft, offset drop-shadow layer from the subject's alpha mask.

    Returns an RGBA layer (size x size) holding a blurred, dimmed, slightly
    offset silhouette in SHADOW_COLOR — composite it onto the background before
    the subject so it reads as casting a delicate shadow on the wall behind.
    """
    blur = max(1.0, size * SHADOW_BLUR)
    off_x = int(round(size * SHADOW_OFFSET_X))
    off_y = int(round(size * SHADOW_OFFSET_Y))
    shadow_a = alpha.point(lambda v: int(v * SHADOW_OPACITY))
    shadow_a = shadow_a.filter(ImageFilter.GaussianBlur(blur))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(shadow_a, (off_x, off_y))
    shadow = Image.new("RGBA", (size, size), SHADOW_COLOR + (0,))
    shadow.putalpha(canvas)
    return shadow


# ── Pipelined execution ─────────────────────────────────────────────────────
#
# Each pass runs in its own thread. Images flow through queues:
#   files → [P1] → q1 → [P1.25] → q125 → [P1.5] → q15 → [P2] → q2 → [P3] → q3 → [P4] → done
#
# As soon as P1 finishes one image, P1.25 can start on it while P1 works on the
# next image. This overlaps Gemini API waits with BiRefNet/CorridorKey GPU work.

_SENTINEL = None  # signals "no more items" on a queue


def _drain(q_in, q_out):
    """Pass all items from one queue to the next until sentinel."""
    while True:
        item = q_in.get()
        if item is _SENTINEL:
            break
        q_out.put(item)


def run_pipeline(files, ratings, regen_from, settings, options):
    """Run all passes concurrently in a pipelined fashion."""
    total = len(files)
    q1 = Queue()     # P1 → P1.25
    q125 = Queue()   # P1.25 → P1.5
    q15 = Queue()    # P1.5 → P2
    q2 = Queue()     # P2 → P3
    q3 = Queue()     # P3 → P4
    errors = []      # collect worker errors

    # ── Pass 1: Upscale (Gemini) ──

    def worker_pass1():
        try:
            if regen_from > 1:
                _mark_pass_skipped(1, "Upscale")
                _log("PASS 1: Skipped (reusing step1 output)")
                for f in files:
                    q1.put(f)
                return

            if not any_wants_upscale(settings, options, files):
                _log("PASS 1: Upscale off for all files — copying originals")
                for i, f in enumerate(files, 1):
                    write_progress(1, "Copy (upscale off)", i, total, f.name)
                    save_img(Image.open(f).convert("RGBA"), STEP1_DIR / f.name)
                    write_progress_file_done(1, f.name)
                    q1.put(f)
                write_progress_pass_done(1)
                return

            from google import genai
            from google.oauth2 import service_account

            if not SERVICE_ACCOUNT_PATH.exists():
                _log("PASS 1: No service_account.json — copying originals")
                for i, f in enumerate(files, 1):
                    write_progress(1, "Copy (no Gemini)", i, total, f.name)
                    img = Image.open(f).convert("RGBA")
                    save_img(img, STEP1_DIR / f.name)
                    write_progress_file_done(1, f.name)
                    q1.put(f)
                write_progress_pass_done(1)
                return

            credentials = service_account.Credentials.from_service_account_file(
                str(SERVICE_ACCOUNT_PATH),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = genai.Client(
                vertexai=True, project=GEMINI_PROJECT,
                location=GEMINI_LOCATION, credentials=credentials,
            )
            _log(f"P1: Gemini connected ({GEMINI_MODEL})")

            for i, f in enumerate(files, 1):
                fname = f.name
                if not effective_upscale(settings, options, fname):
                    write_progress(1, "Copy (upscale off)", i, total, fname)
                    save_img(Image.open(f).convert("RGBA"), STEP1_DIR / fname)
                    write_progress_file_done(1, fname)
                    q1.put(f)
                    continue

                write_progress(1, "Enhance (Gemini)", i, total, fname)
                img = Image.open(f).convert("RGBA")
                # aspect_ratio="1:1" restores the squared canvas (Nano Banana no
                # longer squares portraits by default). image_size is forward-
                # compat; the real 2K comes from the Lanczos pass below.
                result = gemini_enhance(client, img.convert("RGB"),
                                        image_size="2K", aspect_ratio="1:1")
                if result is not None:
                    result = upscale_long_edge(result)
                    _log(f"[P1 {i}/{total}] {fname}: {img.size[0]}x{img.size[1]} → {result.size[0]}x{result.size[1]}")
                    save_img(result, STEP1_DIR / fname)
                    write_progress_file_done(1, fname)
                else:
                    _log(f"[P1 {i}/{total}] {fname}: failed, copying original")
                    save_img(img, STEP1_DIR / fname)
                    write_progress_file_done(1, fname, "fallback")
                q1.put(f)

            write_progress_pass_done(1)
            _log("P1: Done")
        except Exception as e:
            errors.append(("P1", e))
            _log(f"P1 ERROR: {e}")
        finally:
            q1.put(_SENTINEL)

    # ── Pass 1.25: Green Background Substitution (BiRefNet) ──

    def worker_pass125():
        try:
            if regen_from > 1.25:
                _mark_pass_skipped(1.25, "Green BG")
                _log("PASS 1.25: Skipped (reusing step1 output)")
                _drain(q1, q125)
                return

            from rembg import remove, new_session
            _log("P1.25: Loading BiRefNet for green BG substitution...")
            session = new_session("birefnet-portrait")
            _log("P1.25: BiRefNet loaded")

            i = 0
            while True:
                f = q1.get()
                if f is _SENTINEL:
                    break
                i += 1
                fname = f.name
                s1_path = STEP1_DIR / fname
                if not s1_path.exists():
                    write_progress(1.25, "Green BG", i, total, fname, "skipped")
                    write_progress_file_done(1.25, fname, "skipped")
                    q125.put(f)
                    continue

                write_progress(1.25, "Green BG", i, total, fname)
                img = Image.open(s1_path).convert("RGBA")

                # Use BiRefNet to get person mask
                img_no_bg = remove(
                    img, session=session, alpha_matting=True,
                    alpha_matting_foreground_threshold=230,
                    alpha_matting_background_threshold=20,
                    alpha_matting_erode_size=6,
                )
                _, _, _, alpha = img_no_bg.split()

                # Composite person over chroma green background
                green_bg = Image.new("RGBA", img.size, CHROMA_GREEN + (255,))
                person = img.copy()
                person.putalpha(alpha)
                result = Image.alpha_composite(green_bg, person)

                save_img(result, s1_path)  # Overwrite step1 in-place
                _log(f"[P1.25 {i}/{total}] {fname}: green BG applied")
                write_progress_file_done(1.25, fname)
                q125.put(f)

            del session
            write_progress_pass_done(1.25)
            _log("P1.25: Done, BiRefNet unloaded")
        except Exception as e:
            errors.append(("P1.25", e))
            _log(f"P1.25 ERROR: {e}")
        finally:
            q125.put(_SENTINEL)

    # ── Pass 1.5: Canvas Extend (Gemini) ──

    def worker_pass15():
        try:
            if regen_from > 1.5:
                _mark_pass_skipped(1.5, "Canvas Extend")
                _log("PASS 1.5: Skipped (reusing step1 output)")
                _drain(q125, q15)
                return

            if not any_wants_extend(settings, options, files):
                _mark_pass_skipped(1.5, "Canvas Extend")
                _log("PASS 1.5: Canvas extend off for all files — skipping")
                _drain(q125, q15)
                return

            if not SERVICE_ACCOUNT_PATH.exists():
                _mark_pass_skipped(1.5, "Canvas Extend")
                _log("PASS 1.5: No service_account.json — skipping")
                _drain(q125, q15)
                return

            client = get_gemini_client()
            _log("P1.5: Gemini canvas extension connected")

            extend_prompt = (
                "Re-frame this portrait as a SQUARE. The head and shoulders should "
                "FILL most of the frame — the top of the hair sits near the top "
                "edge and the shoulders span most of the width. Keep only a small "
                "even margin around the person; do NOT zoom out or shrink them.\n\n"
                "1. EXTEND THE BODY: The torso and shoulders are cropped at the "
                "bottom edge. GENERATE the natural continuation of the shoulders, "
                "chest and clothing downward so the body does NOT end in a flat "
                "straight cut or a tapered point. Keep the face, hair, expression "
                "and features EXACTLY identical and undistorted — same person.\n\n"
                "2. UNIFORM CHROMA GREEN BACKGROUND — CRITICAL: Replace the entire "
                "background (everything that is NOT the person) with a single flat "
                "uniform chroma key green color, RGB (0, 177, 64) — pure bright green. "
                "The background must be ONE solid pure green color, edge to edge, "
                "with no hard rectangular edges around the body.\n\n"
                "DO NOT use white, gray, or any other color. DO NOT add gradients, "
                "lighting, shadows, vignettes, or texture to the background. DO NOT "
                "preserve any original background elements.\n\n"
                "Output only the extended photo with uniform chroma green background."
            )

            i = 0
            while True:
                f = q125.get()
                if f is _SENTINEL:
                    break
                i += 1
                fname = f.name
                if not effective_extend(settings, options, fname):
                    write_progress(1.5, "Canvas Extend (off)", i, total, fname, "skipped")
                    write_progress_file_done(1.5, fname, "skipped")
                    q15.put(f)
                    continue
                s1_path = STEP1_DIR / fname
                if not s1_path.exists():
                    write_progress(1.5, "Canvas Extend (Gemini)", i, total, fname, "skipped")
                    write_progress_file_done(1.5, fname, "skipped")
                    q15.put(f)
                    continue

                write_progress(1.5, "Canvas Extend (Gemini)", i, total, fname)
                img = Image.open(s1_path).convert("RGB")
                # Ask Gemini to re-frame square and genuinely outpaint the
                # torso/shoulders that were cropped, on uniform green. Feeding
                # the image directly (no pre-pad) makes the model generate new
                # body; pre-padding green just makes it taper the body into a cut.
                try:
                    result = gemini_enhance(client, img, prompt=extend_prompt,
                                            image_size="2K", aspect_ratio="1:1")
                except Exception as e:
                    _log(f"[P1.5 {i}/{total}] {fname}: error {str(e)[:80]} — keeping upscaled original")
                    save_img(upscale_long_edge(img), s1_path)
                    write_progress_file_done(1.5, fname, "error")
                    q15.put(f)
                    continue

                if result is not None:
                    result = upscale_long_edge(result)
                    _log(f"[P1.5 {i}/{total}] {fname}: {img.size[0]}x{img.size[1]} → {result.size[0]}x{result.size[1]}")
                    save_img(result, s1_path)
                    write_progress_file_done(1.5, fname)
                else:
                    _log(f"[P1.5 {i}/{total}] {fname}: AI extend failed, keeping upscaled original")
                    save_img(upscale_long_edge(img), s1_path)
                    write_progress_file_done(1.5, fname, "fallback")
                q15.put(f)

            write_progress_pass_done(1.5)
            _log("P1.5: Done")
        except Exception as e:
            errors.append(("P1.5", e))
            _log(f"P1.5 ERROR: {e}")
        finally:
            q15.put(_SENTINEL)

    # ── Pass 2: Green Screen Keying (CorridorKey) ──

    def worker_pass2():
        try:
            if regen_from > 2:
                _mark_pass_skipped(2, "Key")
                _log("PASS 2: Skipped (reusing step2 output)")
                _drain(q15, q2)
                return

            ck_path = str(CORRIDORKEY_DIR)
            if ck_path not in sys.path:
                sys.path.insert(0, ck_path)

            from CorridorKeyModule.inference_engine import CorridorKeyEngine
            from device_utils import detect_best_device

            device = detect_best_device()
            checkpoint = CORRIDORKEY_DIR / "CorridorKeyModule" / "checkpoints" / "CorridorKey.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(f"CorridorKey checkpoint not found: {checkpoint}")
            # img_size=512 for MPS/CPU feasibility; mixed_precision=False for MPS compat
            ck_img_size = 2048 if device == "cuda" else 512
            _log(f"P2: Loading CorridorKey engine (device={device}, img_size={ck_img_size})...")
            engine = CorridorKeyEngine(
                checkpoint_path=str(checkpoint),
                device=device,
                img_size=ck_img_size,
                mixed_precision=(device == "cuda"),
            )
            _log("P2: CorridorKey loaded")

            i = 0
            while True:
                f = q15.get()
                if f is _SENTINEL:
                    break
                i += 1
                fname = f.name
                stem = Path(fname).stem
                s1_path = STEP1_DIR / fname
                if not s1_path.exists():
                    _log(f"[P2 {i}/{total}] {fname}: skip (no step1)")
                    write_progress(2, "Key (CorridorKey)", i, total, fname, "skipped")
                    write_progress_file_done(2, fname, "skipped")
                    q2.put(f)
                    continue

                write_progress(2, "Key (CorridorKey)", i, total, fname)
                img_pil = Image.open(s1_path).convert("RGB")
                img_np = np.array(img_pil).astype(np.float32) / 255.0

                # Chroma threshold: foreground where green doesn't dominate
                r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
                mask = (g - np.maximum(r, b) < 0.1).astype(np.float32)

                ck_result = engine.process_frame(
                    image=img_np,
                    mask_linear=mask,
                    input_is_linear=False,
                    despill_strength=1.0,
                    auto_despeckle=True,
                    despeckle_size=400,
                    generate_comp=False,
                    post_process_on_gpu=(device != "cpu"),
                )

                fg = ck_result["fg"]
                alpha = ck_result["alpha"]
                rgba_u8 = np.clip(np.concatenate([fg, alpha], axis=2) * 255, 0, 255).astype(np.uint8)
                img_out = Image.fromarray(rgba_u8, "RGBA")

                img_out.save(STEP2_DIR / (stem + ".png"), "PNG")
                _log(f"[P2 {i}/{total}] {fname}: done (CorridorKey)")
                write_progress_file_done(2, fname)
                q2.put(f)

            del engine
            write_progress_pass_done(2)
            _log("P2: Done")
        except Exception as e:
            errors.append(("P2", e))
            _log(f"P2 ERROR: {e}")
        finally:
            q2.put(_SENTINEL)
            # Block forever — PyTorch's C++ thread-local destructors crash
            # (SIGSEGV in take_gil) during pthread_exit. As a daemon thread,
            # this will be killed cleanly at interpreter shutdown.
            threading.Event().wait()

    # ── Pass 3: Color Adjustments (per-image ratings, no B&W) ──

    def worker_pass3():
        try:
            if regen_from > 3:
                _mark_pass_skipped(3, "Adjust")
                _log("PASS 3: Skipped (reusing step3 output)")
                _drain(q2, q3)
                return

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

            i = 0
            while True:
                f = q2.get()
                if f is _SENTINEL:
                    break
                i += 1
                fname = f.name
                stem = Path(fname).stem
                s2_path = STEP2_DIR / (stem + ".png")
                if not s2_path.exists():
                    _log(f"[P3 {i}/{total}] {fname}: skip (no step2)")
                    write_progress(3, "Adjust", i, total, fname, "skipped")
                    write_progress_file_done(3, fname, "skipped")
                    q3.put(f)
                    continue

                write_progress(3, "Adjust", i, total, fname)
                img = Image.open(s2_path).convert("RGBA")
                _, _, _, a = img.split()
                a_np = np.array(a)

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
                adj_str = ""
                if has_adj:
                    adj_str = f" [L={adj_vals[0]} C={adj_vals[1]} D={adj_vals[2]} H={adj_vals[3]} S={adj_vals[4]}]"
                _log(f"[P3 {i}/{total}] {fname}: {kind}{adj_str}")

                img_adj.save(STEP3_DIR / (stem + ".png"), "PNG")
                write_progress_file_done(3, fname)
                q3.put(f)

            write_progress_pass_done(3)
            _log("P3: Done")
        except Exception as e:
            errors.append(("P3", e))
            _log(f"P3 ERROR: {e}")
        finally:
            q3.put(_SENTINEL)

    # ── Pass 4: Leafs Background Composite ──

    def worker_pass4():
        try:
            bg_img = Image.open(background_path(settings)).convert("RGBA")
            # Center-crop the background to a square so resizing won't distort it.
            bw_, bh_ = bg_img.size
            if bw_ != bh_:
                bside = min(bw_, bh_)
                bl = (bw_ - bside) // 2
                bt = (bh_ - bside) // 2
                bg_img = bg_img.crop((bl, bt, bl + bside, bt + bside))
            # Reference colour stats = the background (per RGB channel).
            ref_stats = compute_ref_stats(np.array(bg_img.convert("RGB")))
            i = 0
            while True:
                f = q3.get()
                if f is _SENTINEL:
                    break
                i += 1
                fname = f.name
                stem = Path(fname).stem
                s3_path = STEP3_DIR / (stem + ".png")
                if not s3_path.exists():
                    write_progress(4, "Background", i, total, fname, "skipped")
                    write_progress_file_done(4, fname, "skipped")
                    continue

                write_progress(4, "Background", i, total, fname)
                img = Image.open(s3_path).convert("RGBA")
                w, h = img.size
                if w != h:
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    img = img.crop((left, top, left + side, top + side))

                # Align the person's per-channel histogram to the background
                # before compositing, so the subject sits in the foliage palette.
                strength = bg_match_strength(effective_bg_match_amount(settings, options, fname))
                if any(s > 0 for s in strength):
                    r, g, b, a = img.split()
                    fg_mask = np.array(a) > 0
                    matched = match_histogram(
                        np.array(Image.merge("RGB", (r, g, b))),
                        fg_mask, ref_stats, strength,
                    )
                    img = Image.merge("RGBA", (*Image.fromarray(matched).split(), a))

                size = img.size[0]
                bg = bg_img.copy().resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")
                # Cast a delicate drop shadow onto the background before the subject.
                if SHADOW_OPACITY > 0:
                    bg = Image.alpha_composite(bg, make_drop_shadow(img.split()[3], size))
                result = Image.alpha_composite(bg, img)
                save_img(result, STEP4_DIR / fname)
                _log(f"[P4 {i}/{total}] {fname}: done")
                write_progress_file_done(4, fname)

            write_progress_pass_done(4)
            _log("P4: Done")
        except Exception as e:
            errors.append(("P4", e))
            _log(f"P4 ERROR: {e}")

    # ── Launch pipeline ──

    _log("=" * 60)
    _log("PIPELINE: 6 stages running concurrently")
    _log("  P1:Upscale → P1.25:Green BG → P1.5:Extend → P2:Key → P3:Adjust → P4:Background")
    _log("=" * 60)

    p2_thread = threading.Thread(target=worker_pass2, name="P2", daemon=True)
    threads = [
        threading.Thread(target=worker_pass1, name="P1"),
        threading.Thread(target=worker_pass125, name="P1.25"),
        threading.Thread(target=worker_pass15, name="P1.5"),
        p2_thread,
        threading.Thread(target=worker_pass3, name="P3"),
        threading.Thread(target=worker_pass4, name="P4"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        if not t.daemon:
            t.join()
    # P2 is a daemon thread that blocks forever after completing its work
    # to avoid PyTorch's C++ TLS destructor crash. Wait for its sentinel
    # (already sent to q2) by checking that P3 consumed it.
    # P3 and P4 are already joined above, so P2's work is guaranteed done.

    if errors:
        _log(f"\nPipeline finished with {len(errors)} error(s):")
        for name, err in errors:
            _log(f"  {name}: {err}")
    else:
        _log("\nPipeline complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for d in [STEP1_DIR, STEP2_DIR, STEP3_DIR, STEP4_DIR]:
        d.mkdir(exist_ok=True)

    # Parse arguments
    args = sys.argv[1:]
    regenerate = "--regenerate" in args or "--regen" in args
    regen_from = 1  # default: run all passes

    # --regen-from N: skip passes before N, reuse earlier step outputs
    for j, a in enumerate(args):
        if a in ("--regen-from", "--from") and j + 1 < len(args):
            try:
                regen_from = float(args[j + 1])
            except ValueError:
                pass

    # Filter out flags from file targets
    file_targets = [a for a in args if not a.startswith("--") and not a.replace(".", "").isdigit()]

    ratings = load_ratings()
    settings = load_settings()
    options = load_image_options()
    print(f"Run settings: steps={settings['steps']} bg={settings['background']} "
          f"match={settings['bg_match_amount']}")
    prev_ratings = load_prev_ratings()

    extensions = {".webp", ".jpg", ".jpeg", ".png"}
    all_files = sorted(f for f in WEBP_DIR.iterdir() if f.suffix.lower() in extensions)

    if file_targets:
        targets = set(file_targets)
        files = [f for f in all_files if f.name in targets]
        print(f"Processing {len(files)} specified files.\n")
    elif regenerate:
        files = all_files
        print(f"Regenerating ALL {len(files)} images.\n")
    else:
        # Auto-detect: only process files with changed ratings
        files = get_changed_files(ratings, prev_ratings, all_files)
        if not files:
            write_progress_done()
            print("\nNothing to do.")
            return
        print(f"\nProcessing {len(files)} images.\n")

    # Run pipelined passes
    run_pipeline(files, ratings, regen_from, settings, options)

    # Save current ratings as prev for next run's diff
    save_prev_ratings(ratings)

    # Archive run to timestamped folder and update run_info.json
    import datetime
    run_name = datetime.datetime.now().strftime("%y%m%d_%H%M")
    run_dir = BASE_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    import shutil
    for step_dir in [STEP1_DIR, STEP2_DIR, STEP3_DIR, STEP4_DIR]:
        if step_dir.exists():
            dest = run_dir / step_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(step_dir, dest)
    print(f"Archived to {run_name}/")

    # Read previous run_info to shift current -> previous
    run_info_path = BASE_DIR / "run_info.json"
    prev_run = ""
    if run_info_path.exists():
        try:
            old_info = json.loads(run_info_path.read_text())
            prev_run = old_info.get("current", "")
        except Exception:
            pass

    run_info = {
        "current": run_name,
        "previous": prev_run,
    }
    run_info_path.write_text(json.dumps(run_info, indent=2))
    print(f"Updated run_info.json: current={run_name}, previous={prev_run}")

    write_progress_done()
    print("\nAll done.")


if __name__ == "__main__":
    main()
