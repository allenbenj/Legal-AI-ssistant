import os
from pathlib import Path

import pytest

from legal_ai_system.utils import ocr_cache

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


@pytest.fixture(autouse=True)
def clear_cache(tmp_path, monkeypatch):
    # use temp cache directory for tests
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("OCR_CACHE_DIR", str(cache_dir))
    from importlib import reload

    reload(ocr_cache)
    yield
    if cache_dir.exists():
        for p in cache_dir.iterdir():
            p.unlink()
        cache_dir.rmdir()


def _create_image(path: Path, color=(255, 255, 255)) -> None:
    img = Image.new("RGB", (20, 20), color=color)
    img.save(path)


@pytest.mark.unit
@pytest.mark.skipif(
    Image is None or ocr_cache._cache is None,
    reason="Pillow or diskcache not available",
)
def test_ocr_cache_roundtrip(tmp_path):
    img_path = tmp_path / "img.png"
    _create_image(img_path)

    text = "sample-ocr"
    file_hash = ocr_cache.compute_file_hash(img_path)
    assert ocr_cache.get(img_path, 1, file_hash=file_hash) is None

    ocr_cache.set(img_path, 1, text, file_hash=file_hash)
    assert ocr_cache.get(img_path, 1, file_hash=file_hash) == text

    # Changing file invalidates cache key
    _create_image(img_path, color=(0, 0, 0))
    new_hash = ocr_cache.compute_file_hash(img_path)
    assert new_hash != file_hash
    assert ocr_cache.get(img_path, 1, file_hash=new_hash) is None


def test_compute_file_hash_changes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one")
    h1 = ocr_cache.compute_file_hash(f)
    f.write_text("two")
    h2 = ocr_cache.compute_file_hash(f)
    assert h1 != h2
