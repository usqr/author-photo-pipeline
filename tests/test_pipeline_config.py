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


def test_any_wants_helpers():
    s = {"steps": {"upscale": False, "canvas_extend": False, "bw": True, "bg_match": True}}
    files = [type("F", (), {"name": n})() for n in ("a.jpg", "b.jpg")]
    opts = {"b.jpg": {"upscale": True}}
    assert rc.any_wants_upscale(s, opts, files) is True     # b overrides on
    assert rc.any_wants_extend(s, {}, files) is False        # nobody wants extend


def test_load_settings_custom_background(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"background": "custom", "custom_background": "custom_bg.jpg"}))
    monkeypatch.setattr(rc, "SETTINGS_PATH", p)
    s = rc.load_settings()
    assert s["background"] == "custom"
    assert s["custom_background"] == "custom_bg.jpg"
