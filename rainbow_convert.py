#!/usr/bin/env python3
"""
Multi-step author photo pipeline (v6) — memory-safe batched processing.
Runs each step as a separate pass to avoid OOM from loading all models at once.

  Pass 1:    AI upscale (Gemini Nano Banana) → step1_upscaled/
  Pass 1.25: Green background substitution (BiRefNet) → step1_upscaled/ (in-place)
  Pass 1.5:  Canvas extension (Gemini) → step1_upscaled/ (in-place)
  Pass 2:    Green screen keying (CorridorKey) → step2_nobg/
  Pass 3:    B&W conversion (Gemini + per-image ratings) → step3_bw/
  Pass 4:    Rainbow composite → step4_rainbow/

Reads ratings.json for per-image adjustments.
"""

from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from skimage.metrics import structural_similarity as ssim
import cv2
import json
import numpy as np
import sys
import time
import tempfile
import threading
from queue import Queue

BASE_DIR = Path(__file__).parent
WEBP_DIR = BASE_DIR / "webp"
RAINBOW_BG_PATH = BASE_DIR / "rainbow.png"
BW_REF_PATH = BASE_DIR / "bw.png"
RAINBOW_REF_PATH = BASE_DIR / "rainbow_Gokce.jpg"
RATINGS_PATH = BASE_DIR / "ratings.json"
PREV_RATINGS_PATH = BASE_DIR / "prev_ratings.json"
SERVICE_ACCOUNT_PATH = BASE_DIR / "service_account.json"
GEMINI_PROJECT = "gemini-image-generation-492101"
GEMINI_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash-image"

STEP1_DIR = BASE_DIR / "step1_upscaled"
STEP2_DIR = BASE_DIR / "step2_nobg"
STEP3_DIR = BASE_DIR / "step3_bw"
STEP4_DIR = BASE_DIR / "step4_rainbow"
CORRIDORKEY_DIR = BASE_DIR / "CorridorKey"
CHROMA_GREEN = (0, 177, 64)  # Standard broadcast chroma green

TARGET_SIZE = 280
AI_UPSCALE_THRESHOLD = 500

# Pre-compute reference histogram CDF
_bw_ref_rgba = np.array(Image.open(BW_REF_PATH).convert("RGBA"))
_ref_alpha = _bw_ref_rgba[:, :, 3]
_ref_gray = np.array(Image.open(BW_REF_PATH).convert("L"))
_ref_fg_mask = _ref_alpha > 128
_ref_fg = _ref_gray[_ref_fg_mask]
_ref_hist, _ = np.histogram(_ref_fg, bins=256, range=(0, 256))
REF_CDF = np.cumsum(_ref_hist).astype(float)
REF_CDF /= REF_CDF[-1]


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


# ── Pass 1: Upscale (Gemini Nano Banana) ─────────────────────────────────────

def gemini_enhance(client, img_pil, prompt=None):
    """Enhance a portrait image using Gemini image generation.
    Returns enhanced PIL Image or None on failure."""
    from google.genai import types as gtypes
    import time

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

def apply_rating_adjustments(gray_pil, a_np, ratings, fname):
    """Apply per-image rating adjustments to a grayscale image.
    Returns RGBA image with cleaned alpha."""
    r_light = get_rating(ratings, fname, "lightness", 0)
    r_contrast = get_rating(ratings, fname, "contrast", 0)
    r_dark = get_rating(ratings, fname, "dark_areas", 0)
    r_light_areas = get_rating(ratings, fname, "light_areas", 0)
    r_sharp = get_rating(ratings, fname, "sharpness", 0)

    gray_np = np.array(gray_pil)

    # Curves for dark/light areas
    if r_dark != 0 or r_light_areas != 0:
        gray_f = gray_np.astype(np.float32)
        if r_dark < 0:
            lift = abs(r_dark) / 100.0 * 0.8
            m = gray_f < 128
            gray_f[m] = gray_f[m] + lift * (128 - gray_f[m])
        elif r_dark > 0:
            crush = r_dark / 100.0 * 0.7
            m = gray_f < 128
            gray_f[m] = gray_f[m] * (1 - crush)
        if r_light_areas < 0:
            pull = abs(r_light_areas) / 100.0 * 0.7
            m = gray_f > 128
            gray_f[m] = gray_f[m] - pull * (gray_f[m] - 128)
        elif r_light_areas > 0:
            push = r_light_areas / 100.0 * 0.7
            m = gray_f > 128
            gray_f[m] = gray_f[m] + push * (255 - gray_f[m])
        gray_np = np.clip(gray_f, 0, 255).astype(np.uint8)

    gray_pil = Image.fromarray(gray_np)

    # Brightness
    brightness_factor = 1.0 + (r_light / 100.0) * 0.8
    if brightness_factor != 1.0:
        gray_pil = ImageEnhance.Brightness(gray_pil).enhance(max(0.2, brightness_factor))

    # Contrast
    contrast_factor = 1.0 + (r_contrast / 100.0) * 0.8
    if contrast_factor != 1.0:
        gray_pil = ImageEnhance.Contrast(gray_pil).enhance(max(0.2, contrast_factor))

    # Sharpness
    sharpness_factor = 1.0 + (r_sharp / 100.0) * 1.5
    if sharpness_factor != 1.0:
        gray_pil = ImageEnhance.Sharpness(gray_pil).enhance(max(0.0, sharpness_factor))

    has_adj = any(v != 0 for v in [r_light, r_contrast, r_dark, r_light_areas, r_sharp])

    # Clean alpha
    a_float = a_np.astype(np.float32)
    a_clean = np.clip((a_float - 20) * (255.0 / (235 - 20)), 0, 255).astype(np.uint8)

    gray_final = np.array(gray_pil)
    img_bw = Image.merge("RGBA", (
        Image.fromarray(gray_final),
        Image.fromarray(gray_final),
        Image.fromarray(gray_final),
        Image.fromarray(a_clean),
    ))
    return img_bw, has_adj, [r_light, r_contrast, r_dark, r_light_areas, r_sharp]


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


def run_pipeline(files, ratings, regen_from):
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
                write_progress(1, "Enhance (Gemini)", i, total, fname)
                img = Image.open(f).convert("RGBA")
                result = gemini_enhance(client, img.convert("RGB"))
                if result is not None:
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

            if not SERVICE_ACCOUNT_PATH.exists():
                _mark_pass_skipped(1.5, "Canvas Extend")
                _log("PASS 1.5: No service_account.json — skipping")
                _drain(q125, q15)
                return

            client = get_gemini_client()
            _log("P1.5: Gemini canvas extension connected")

            extend_prompt = (
                "Take this portrait photo on a green screen background and extend the "
                "canvas outward by about 15% on every side (left, right, top, bottom). "
                "Generate the missing content naturally — continue the person's body, "
                "hair, clothing seamlessly beyond the original edges. The background "
                "MUST remain the same solid green color throughout. The original photo "
                "should remain in the center, completely unchanged. Keep the person "
                "EXACTLY the same. Output only the extended photo."
            )

            i = 0
            while True:
                f = q125.get()
                if f is _SENTINEL:
                    break
                i += 1
                fname = f.name
                s1_path = STEP1_DIR / fname
                if not s1_path.exists():
                    write_progress(1.5, "Canvas Extend (Gemini)", i, total, fname, "skipped")
                    write_progress_file_done(1.5, fname, "skipped")
                    q15.put(f)
                    continue

                write_progress(1.5, "Canvas Extend (Gemini)", i, total, fname)
                img = Image.open(s1_path).convert("RGB")
                try:
                    result = gemini_enhance(client, img, prompt=extend_prompt)
                except Exception as e:
                    _log(f"[P1.5 {i}/{total}] {fname}: error {str(e)[:80]}")
                    write_progress_file_done(1.5, fname, "error")
                    q15.put(f)
                    continue

                if result is not None:
                    _log(f"[P1.5 {i}/{total}] {fname}: {img.size[0]}x{img.size[1]} → {result.size[0]}x{result.size[1]}")
                    save_img(result, s1_path)
                    write_progress_file_done(1.5, fname)
                else:
                    _log(f"[P1.5 {i}/{total}] {fname}: failed, keeping original")
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

    # ── Pass 3: B&W Conversion (Gemini + adjustments) ──

    def worker_pass3():
        try:
            if regen_from > 3:
                _mark_pass_skipped(3, "B&W")
                _log("PASS 3: Skipped (reusing step3 output)")
                _drain(q2, q3)
                return

            from google import genai
            from google.oauth2 import service_account

            client = None
            if SERVICE_ACCOUNT_PATH.exists():
                credentials = service_account.Credentials.from_service_account_file(
                    str(SERVICE_ACCOUNT_PATH),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                client = genai.Client(
                    vertexai=True, project=GEMINI_PROJECT,
                    location=GEMINI_LOCATION, credentials=credentials,
                )
                _log("P3: Gemini B&W conversion connected")

            bw_prompt = (
                "Convert this portrait photo to high-contrast black and white, "
                "matching the style of the reference image provided. "
                "The result should have: bright whites on skin highlights, "
                "deep rich blacks in hair and dark areas, sharp detail, "
                "and a clean professional look. Keep the person on a transparent/white "
                "background. Keep all details identical — same pose, expression, features. "
                "Output only the B&W image."
            )

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
                    write_progress(3, "B&W (Gemini)", i, total, fname, "skipped")
                    write_progress_file_done(3, fname, "skipped")
                    q3.put(f)
                    continue

                write_progress(3, "B&W (Gemini)", i, total, fname)
                img = Image.open(s2_path).convert("RGBA")
                _, _, _, a = img.split()
                a_np = np.array(a)

                gemini_gray = None
                bw_method = "gemini"
                if client is not None:
                    result = gemini_enhance(client, img.convert("RGB"), prompt=bw_prompt)
                    if result is not None:
                        gemini_gray = result.convert("L")

                if gemini_gray is None:
                    bw_method = "local"
                    gray_np = np.array(img.convert("L"))
                    fg_mask = a_np > 128
                    if fg_mask.any():
                        src_fg = gray_np[fg_mask]
                        src_hist, _ = np.histogram(src_fg, bins=256, range=(0, 256))
                        src_cdf = np.cumsum(src_hist).astype(float)
                        src_cdf /= src_cdf[-1]
                        mapping = np.zeros(256, dtype=np.uint8)
                        for j in range(256):
                            k = np.searchsorted(REF_CDF, src_cdf[j])
                            mapping[j] = min(k, 255)
                        gray_np = mapping[gray_np]
                    gemini_gray = Image.fromarray(gray_np)
                    gemini_gray = ImageOps.autocontrast(gemini_gray, cutoff=0.5)

                img_bw, has_adj, adj_vals = apply_rating_adjustments(gemini_gray, a_np, ratings, fname)
                adj_str = ""
                if has_adj:
                    adj_str = f" [L={adj_vals[0]} C={adj_vals[1]} D={adj_vals[2]} H={adj_vals[3]} S={adj_vals[4]}]"
                _log(f"[P3 {i}/{total}] {fname}: {bw_method}{adj_str}")

                img_bw.save(STEP3_DIR / (stem + ".png"), "PNG")
                status = "done" if bw_method == "gemini" else "fallback"
                write_progress_file_done(3, fname, status)
                q3.put(f)

            write_progress_pass_done(3)
            _log("P3: Done")
        except Exception as e:
            errors.append(("P3", e))
            _log(f"P3 ERROR: {e}")
        finally:
            q3.put(_SENTINEL)

    # ── Pass 4: Rainbow Composite ──

    def worker_pass4():
        try:
            bg_img = Image.open(RAINBOW_BG_PATH).convert("RGBA")
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
                    write_progress(4, "Rainbow", i, total, fname, "skipped")
                    write_progress_file_done(4, fname, "skipped")
                    continue

                write_progress(4, "Rainbow", i, total, fname)
                img = Image.open(s3_path).convert("RGBA")
                w, h = img.size
                if w != h:
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    img = img.crop((left, top, left + side, top + side))

                size = img.size[0]
                bg = bg_img.copy().resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")
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
    _log("  P1:Upscale → P1.25:Green BG → P1.5:Extend → P2:Key → P3:B&W → P4:Rainbow")
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
    run_pipeline(files, ratings, regen_from)

    # SSIM
    gokce_bw = STEP3_DIR / "gokce-f3ee914ba0 (1).png"
    gokce_rainbow = STEP4_DIR / "gokce-f3ee914ba0 (1).jpg"
    if gokce_bw.exists() and gokce_rainbow.exists():
        print("--- Gokce SSIM ---")
        for path, ref, label in [
            (gokce_bw, BW_REF_PATH, "B&W"),
            (gokce_rainbow, RAINBOW_REF_PATH, "Rainbow"),
        ]:
            img_arr = np.array(Image.open(path).convert("L"))
            ref_arr = np.array(Image.open(ref).convert("L"))
            if img_arr.shape != ref_arr.shape:
                img_arr = cv2.resize(img_arr, (ref_arr.shape[1], ref_arr.shape[0]))
            print(f"  SSIM ({label}): {ssim(ref_arr, img_arr):.4f}")

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
