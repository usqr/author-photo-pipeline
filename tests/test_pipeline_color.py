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
