#!/usr/bin/env python3
"""
Multi-step author photo pipeline (v5) — memory-safe batched processing.
Runs each step as a separate pass to avoid OOM from loading all models at once.

  Pass 1: AI upscale (EDSR) → step1_upscaled/
  Pass 2: Background removal (BiRefNet portrait) → step2_nobg/
  Pass 3: B&W conversion (histogram-matched + per-image ratings) → step3_bw/
  Pass 4: Rainbow composite → step4_rainbow/

Reads ratings.json for per-image adjustments.
"""

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.metrics import structural_similarity as ssim
import cv2
import json
import numpy as np
import subprocess
import sys
import tempfile

BASE_DIR = Path(__file__).parent
WEBP_DIR = BASE_DIR / "webp"
RAINBOW_BG_PATH = BASE_DIR / "rainbow.png"
BW_REF_PATH = BASE_DIR / "bw.png"
RAINBOW_REF_PATH = BASE_DIR / "rainbow_Gokce.jpg"
RATINGS_PATH = BASE_DIR / "ratings.json"

STEP1_DIR = BASE_DIR / "step1_upscaled"
STEP2_DIR = BASE_DIR / "step2_nobg"
STEP3_DIR = BASE_DIR / "step3_bw"
STEP4_DIR = BASE_DIR / "step4_rainbow"

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


def write_progress(pass_num, pass_name, current, total, filename=""):
    """Write progress to JSON file for compare page to poll."""
    data = {
        "pass": pass_num,
        "pass_name": pass_name,
        "current": current,
        "total": total,
        "filename": filename,
        "done": False,
    }
    PROGRESS_PATH.write_text(json.dumps(data))


def write_progress_done():
    data = {"done": True, "pass": 4, "pass_name": "Complete", "current": 0, "total": 0, "filename": ""}
    PROGRESS_PATH.write_text(json.dumps(data))


def load_ratings():
    if RATINGS_PATH.exists():
        with open(RATINGS_PATH) as f:
            ratings = json.load(f)
        print(f"Loaded ratings for {len(ratings)} images")
        return ratings
    return {}


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


# ── Pass 1: Upscale ──────────────────────────────────────────────────────────

def run_pass1(files, ratings):
    """Upscale pass — loads EDSR model only."""
    from super_image import EdsrModel, ImageLoader as IL

    print("Loading EDSR model...")
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)
    print("EDSR loaded.\n")

    for i, f in enumerate(files, 1):
        fname = f.name
        print(f"[{i}/{len(files)}] Upscale: {fname}")
        write_progress(1, "Upscale", i, len(files), fname)

        img = Image.open(f).convert("RGBA")
        w, h = img.size
        min_dim = min(w, h)
        detail_rating = get_rating(ratings, fname, "detail", 0)
        needs_detail = detail_rating > 0

        if min_dim < AI_UPSCALE_THRESHOLD or needs_detail:
            # Denoise very small images
            if min_dim < 250:
                rgb_np = np.array(img.convert("RGB"))
                denoised = cv2.fastNlMeansDenoisingColored(rgb_np, None, 6, 6, 7, 21)
                img = Image.fromarray(denoised).convert("RGBA")
                print(f"  Denoised {w}x{h}")

            # AI upscale
            print(f"  AI upscale {w}x{h} → ", end="", flush=True)
            rgb = img.convert("RGB")
            inputs = IL.load_image(rgb)
            preds = model(inputs)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp = tf.name
            IL.save_image(preds, tmp)
            result = Image.open(tmp).convert("RGBA")
            if img.mode == "RGBA":
                a = img.split()[3]
                a_up = a.resize(result.size, Image.LANCZOS)
                r, g, b, _ = result.split()
                result = Image.merge("RGBA", (r, g, b, a_up))
            Path(tmp).unlink(missing_ok=True)
            img = result
            w2, h2 = img.size
            print(f"{w2}x{h2}")

            # Bilateral smooth for small source images
            if min_dim < AI_UPSCALE_THRESHOLD:
                rgb_np = np.array(img.convert("RGB"))
                smoothed = cv2.bilateralFilter(rgb_np, d=9, sigmaColor=75, sigmaSpace=75)
                result = Image.fromarray(smoothed).convert("RGBA")
                if img.mode == "RGBA":
                    _, _, _, a = img.split()
                    r, g, b, _ = result.split()
                    result = Image.merge("RGBA", (r, g, b, a))
                img = result
                print(f"  Bilateral smooth")

            # LANCZOS for remaining scale
            work_size = TARGET_SIZE * 2
            if min(img.size) < work_size:
                scale = work_size / min(img.size)
                nw, nh = int(img.size[0] * scale), int(img.size[1] * scale)
                img = img.resize((nw, nh), Image.LANCZOS)
                print(f"  LANCZOS → {nw}x{nh}")
        else:
            work_size = TARGET_SIZE * 2
            if min_dim < work_size:
                scale = work_size / min_dim
                nw, nh = int(w * scale), int(h * scale)
                img = img.resize((nw, nh), Image.LANCZOS)

        save_img(img, STEP1_DIR / fname)

    # Free model memory
    del model
    print("\nPass 1 done. EDSR unloaded.\n")


# ── Pass 2: Background removal ───────────────────────────────────────────────

def run_pass2(files):
    """Background removal pass — loads BiRefNet only."""
    from rembg import remove, new_session

    print("Loading BiRefNet portrait model...")
    session = new_session("birefnet-portrait")
    print("BiRefNet loaded.\n")

    for i, f in enumerate(files, 1):
        fname = f.name
        s1_path = STEP1_DIR / fname
        if not s1_path.exists():
            print(f"[{i}/{len(files)}] Skip (no step1): {fname}")
            continue

        print(f"[{i}/{len(files)}] BG remove: {fname}")
        write_progress(2, "BG Remove", i, len(files), fname)
        img = Image.open(s1_path).convert("RGBA")
        img_no_bg = remove(
            img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=6,
        )
        out_path = STEP2_DIR / (Path(fname).stem + ".png")
        img_no_bg.save(out_path, "PNG")

    del session
    print("\nPass 2 done. BiRefNet unloaded.\n")


# ── Pass 3: B&W conversion ───────────────────────────────────────────────────

def run_pass3(files, ratings):
    """B&W conversion with per-image adjustments. No heavy models needed."""

    for i, f in enumerate(files, 1):
        fname = f.name
        stem = Path(fname).stem
        s2_path = STEP2_DIR / (stem + ".png")
        if not s2_path.exists():
            print(f"[{i}/{len(files)}] Skip (no step2): {fname}")
            continue

        print(f"[{i}/{len(files)}] B&W: {fname}", end="")
        write_progress(3, "B&W Convert", i, len(files), fname)

        img = Image.open(s2_path).convert("RGBA")
        _, _, _, a = img.split()
        gray_np = np.array(img.convert("L"))
        a_np = np.array(a)
        fg_mask = a_np > 128

        # Histogram match
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

        # Per-image ratings
        r_light = get_rating(ratings, fname, "lightness", 0)
        r_contrast = get_rating(ratings, fname, "contrast", 0)
        r_dark = get_rating(ratings, fname, "dark_areas", 0)
        r_light_areas = get_rating(ratings, fname, "light_areas", 0)
        r_sharp = get_rating(ratings, fname, "sharpness", 0)

        # Curves for dark/light areas — strong scaling (0.7-0.8 max effect)
        if r_dark != 0 or r_light_areas != 0:
            gray_f = gray_np.astype(np.float32)
            if r_dark < 0:
                # Too much black → lift shadows aggressively
                lift = abs(r_dark) / 100.0 * 0.8
                m = gray_f < 128
                gray_f[m] = gray_f[m] + lift * (128 - gray_f[m])
            elif r_dark > 0:
                # Needs deeper blacks → crush shadows
                crush = r_dark / 100.0 * 0.7
                m = gray_f < 128
                gray_f[m] = gray_f[m] * (1 - crush)
            if r_light_areas < 0:
                # Too blown out → pull highlights down
                pull = abs(r_light_areas) / 100.0 * 0.7
                m = gray_f > 128
                gray_f[m] = gray_f[m] - pull * (gray_f[m] - 128)
            elif r_light_areas > 0:
                # Needs brighter whites → push highlights up
                push = r_light_areas / 100.0 * 0.7
                m = gray_f > 128
                gray_f[m] = gray_f[m] + push * (255 - gray_f[m])
            gray_np = np.clip(gray_f, 0, 255).astype(np.uint8)

        gray_pil = Image.fromarray(gray_np)
        gray_pil = ImageOps.autocontrast(gray_pil, cutoff=0.5)

        # Brightness: base 1.1, scale ±0.8 (range 0.3 to 1.9)
        brightness_factor = 1.1 + (r_light / 100.0) * 0.8
        gray_pil = ImageEnhance.Brightness(gray_pil).enhance(max(0.2, brightness_factor))

        # Contrast: base 1.0, scale ±0.8 (range 0.2 to 1.8)
        contrast_factor = 1.0 + (r_contrast / 100.0) * 0.8
        if contrast_factor != 1.0:
            gray_pil = ImageEnhance.Contrast(gray_pil).enhance(max(0.2, contrast_factor))

        # Sharpness: base 1.15, scale ±1.5 (range ~0 to 2.65)
        sharpness_factor = 1.15 + (r_sharp / 100.0) * 1.5
        gray_pil = ImageEnhance.Sharpness(gray_pil).enhance(max(0.0, sharpness_factor))

        has_adj = any(v != 0 for v in [r_light, r_contrast, r_dark, r_light_areas, r_sharp])
        if has_adj:
            print(f" [L={r_light} C={r_contrast} D={r_dark} H={r_light_areas} S={r_sharp}]")
        else:
            print()

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
        img_bw.save(STEP3_DIR / (stem + ".png"), "PNG")

    print("\nPass 3 done.\n")


# ── Pass 4: Rainbow composite ────────────────────────────────────────────────

def run_pass4(files):
    bg_img = Image.open(RAINBOW_BG_PATH).convert("RGBA")

    for i, f in enumerate(files, 1):
        fname = f.name
        stem = Path(fname).stem
        s3_path = STEP3_DIR / (stem + ".png")
        if not s3_path.exists():
            continue

        img = Image.open(s3_path).convert("RGBA")
        w, h = img.size
        # Center-crop to square (no downscale)
        if w != h:
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))

        # Scale rainbow background to match image size (don't downscale the image)
        size = img.size[0]
        bg = bg_img.copy().resize((size, size), Image.LANCZOS).convert("RGBA")
        result = Image.alpha_composite(bg, img)

        save_img(result, STEP4_DIR / fname)
        write_progress(4, "Rainbow", i, len(files), fname)

    print("Pass 4 done.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for d in [STEP1_DIR, STEP2_DIR, STEP3_DIR, STEP4_DIR]:
        d.mkdir(exist_ok=True)

    ratings = load_ratings()

    extensions = {".webp", ".jpg", ".jpeg", ".png"}
    all_files = sorted(f for f in WEBP_DIR.iterdir() if f.suffix.lower() in extensions)

    if len(sys.argv) > 1:
        targets = set(sys.argv[1:])
        files = [f for f in all_files if f.name in targets]
        print(f"Processing {len(files)} specified files.\n")
    else:
        files = all_files
        print(f"Processing all {len(files)} images.\n")

    print("=" * 60)
    print("PASS 1: Upscale (EDSR)")
    print("=" * 60)
    run_pass1(files, ratings)

    print("=" * 60)
    print("PASS 2: Background removal (BiRefNet)")
    print("=" * 60)
    run_pass2(files)

    print("=" * 60)
    print("PASS 3: B&W conversion")
    print("=" * 60)
    run_pass3(files, ratings)

    print("=" * 60)
    print("PASS 4: Rainbow composite")
    print("=" * 60)
    run_pass4(files)

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

    write_progress_done()
    print("\nAll done.")


if __name__ == "__main__":
    main()
