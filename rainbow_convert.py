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
from PIL import Image, ImageEnhance, ImageOps
from skimage.metrics import structural_similarity as ssim
import cv2
import json
import numpy as np
import sys
import tempfile

BASE_DIR = Path(__file__).parent
WEBP_DIR = BASE_DIR / "webp"
RAINBOW_BG_PATH = BASE_DIR / "rainbow.png"
BW_REF_PATH = BASE_DIR / "bw.png"
RAINBOW_REF_PATH = BASE_DIR / "rainbow_Gokce.jpg"
RATINGS_PATH = BASE_DIR / "ratings.json"
PREV_RATINGS_PATH = BASE_DIR / "prev_ratings.json"

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


# ── Pass 1: Upscale ──────────────────────────────────────────────────────────

def ai_upscale(img_rgb, model, IL):
    """Run a super_image model on an RGB PIL image, return RGBA result."""
    inputs = IL.load_image(img_rgb)
    preds = model(inputs)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    IL.save_image(preds, tmp)
    result = Image.open(tmp).convert("RGBA")
    Path(tmp).unlink(missing_ok=True)
    return result


def run_pass1(files):
    """Upscale pass — HAN 4x for small images only (<= 500px).
    Larger images skip AI upscaling (too much memory)."""
    from super_image import HanModel, ImageLoader as IL

    # Split: small images get HAN 4x, everything else skips
    small_files = []
    skip_files = []

    for f in files:
        img = Image.open(f)
        min_dim = min(img.size)
        if min_dim <= AI_UPSCALE_THRESHOLD:
            small_files.append(f)
        else:
            skip_files.append(f)
        img.close()

    total = len(files)
    done = 0

    # HAN 4x for small images
    if small_files:
        print(f"Loading HAN 4x for {len(small_files)} small images...")
        model = HanModel.from_pretrained("eugenesiow/han", scale=4)
        for f in small_files:
            done += 1
            fname = f.name
            print(f"[{done}/{total}] Upscale: {fname}")
            write_progress(1, "Upscale (HAN 4x)", done, total, fname)
            img = Image.open(f).convert("RGBA")
            w, h = img.size
            min_dim = min(w, h)
            if min_dim < 250:
                rgb_np = np.array(img.convert("RGB"))
                denoised = cv2.fastNlMeansDenoisingColored(rgb_np, None, 6, 6, 7, 21)
                img = Image.fromarray(denoised).convert("RGBA")
                print(f"  Denoised {w}x{h}")
            print(f"  HAN 4x {w}x{h} → ", end="", flush=True)
            result = ai_upscale(img.convert("RGB"), model, IL)
            if img.mode == "RGBA":
                a = img.split()[3]
                a_up = a.resize(result.size, Image.LANCZOS)
                r, g, b, _ = result.split()
                result = Image.merge("RGBA", (r, g, b, a_up))
            img = result
            print(f"{img.size[0]}x{img.size[1]}")
            if min_dim < 300:
                rgb_np = np.array(img.convert("RGB"))
                smoothed = cv2.bilateralFilter(rgb_np, d=9, sigmaColor=75, sigmaSpace=75)
                sm = Image.fromarray(smoothed).convert("RGBA")
                _, _, _, a = img.split()
                r, g, b, _ = sm.split()
                img = Image.merge("RGBA", (r, g, b, a))
                print(f"  Bilateral smooth")
            save_img(img, STEP1_DIR / fname)
        del model
        print("HAN 4x unloaded.\n")

    # No AI upscale — just copy/resize to working size
    for f in skip_files:
        done += 1
        fname = f.name
        print(f"[{done}/{total}] Upscale: {fname} (skip)")
        write_progress(1, "Upscale (skip)", done, total, fname)
        img = Image.open(f).convert("RGBA")
        w, h = img.size
        min_dim = min(w, h)
        work_size = TARGET_SIZE * 2
        if min_dim < work_size:
            scale = work_size / min_dim
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
        save_img(img, STEP1_DIR / fname)

    print("\nPass 1 done.\n")


# ── Pass 1.5: Canvas extension ────────────────────────────────────────────────

CANVAS_PAD = 0.15  # 15% each side


def extend_canvas(img_pil, pad_fraction=CANVAS_PAD):
    """Extend canvas by pad_fraction on each side using reflect + inpaint + gradient blend."""
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]
    pad_x = int(w * pad_fraction)
    pad_y = int(h * pad_fraction)

    # Reflect-pad for initial fill
    padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]

    # Gradient blend mask (soft transition at edges)
    mask = np.zeros((ph, pw), dtype=np.float32)
    for y in range(pad_y):
        alpha = 1.0 - (y / pad_y)
        mask[y, :] = np.maximum(mask[y, :], alpha)
        mask[ph - 1 - y, :] = np.maximum(mask[ph - 1 - y, :], alpha)
    for x in range(pad_x):
        alpha = 1.0 - (x / pad_x)
        mask[:, x] = np.maximum(mask[:, x], alpha)
        mask[:, pw - 1 - x] = np.maximum(mask[:, pw - 1 - x], alpha)

    # Inpaint extended areas for seamless content
    inpaint_mask = (mask > 0.1).astype(np.uint8) * 255
    inpainted = cv2.inpaint(padded, inpaint_mask, inpaintRadius=12, flags=cv2.INPAINT_TELEA)

    # Blend inpainted edges with reflected content
    mask_3ch = np.stack([mask] * 3, axis=-1)
    blended = (inpainted * mask_3ch + padded * (1 - mask_3ch)).astype(np.uint8)

    return Image.fromarray(blended).convert("RGBA")


def run_pass1_5(files):
    """Extend canvas 15% each side on all upscaled images."""
    for i, f in enumerate(files, 1):
        fname = f.name
        s1_path = STEP1_DIR / fname
        if not s1_path.exists():
            continue

        print(f"[{i}/{len(files)}] Extend canvas: {fname}")
        write_progress(1.5, "Canvas Extend", i, len(files), fname)

        img = Image.open(s1_path).convert("RGBA")
        extended = extend_canvas(img)
        save_img(extended, s1_path)  # overwrite step1 output

    print("\nPass 1.5 done.\n")


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
    prev_ratings = load_prev_ratings()

    extensions = {".webp", ".jpg", ".jpeg", ".png"}
    all_files = sorted(f for f in WEBP_DIR.iterdir() if f.suffix.lower() in extensions)

    if len(sys.argv) > 1:
        targets = set(sys.argv[1:])
        files = [f for f in all_files if f.name in targets]
        print(f"Processing {len(files)} specified files.\n")
    else:
        # Auto-detect: only process files with changed ratings
        files = get_changed_files(ratings, prev_ratings, all_files)
        if not files:
            write_progress_done()
            print("\nNothing to do.")
            return
        print(f"\nProcessing {len(files)} images.\n")

    print("=" * 60)
    print("PASS 1: Upscale (EDSR)")
    print("=" * 60)
    run_pass1(files)

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
