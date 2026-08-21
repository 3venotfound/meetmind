import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


CV_DIR = Path(__file__).resolve().parents[2] / "cv"


def load_module(filename, fake_modules):
    name = f"meetmind_cv_test_{filename}_{id(fake_modules)}"
    spec = importlib.util.spec_from_file_location(name, CV_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


class Capture:
    def __init__(self, opened=True, readable=True):
        self.opened = opened
        self.readable = readable

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 1

    def set(self, prop, value):
        return True

    def read(self):
        return self.readable, object() if self.readable else None

    def release(self):
        pass


def fake_cv2(capture=None, imread=None, imwrite=True):
    module = types.ModuleType("cv2")
    module.CAP_PROP_FPS = 1
    module.CAP_PROP_FRAME_COUNT = 2
    module.CAP_PROP_POS_FRAMES = 3
    module.VideoCapture = lambda path: capture or Capture()
    module.imread = lambda path: imread
    module.imwrite = lambda path, image: imwrite
    module.COLOR_BGR2RGB = 1
    module.cvtColor = lambda image, code: image
    return module


class CVSafetyTests(unittest.TestCase):
    def test_video_open_failure(self):
        module = load_module("video_processor.py", {"cv2": fake_cv2(Capture(opened=False))})
        with self.assertRaisesRegex(OSError, "Could not open"):
            module.extract_frames("bad.webm")

    def test_frame_write_failure(self):
        module = load_module("video_processor.py", {"cv2": fake_cv2(imwrite=False)})
        with tempfile.TemporaryDirectory() as output_directory:
            with self.assertRaisesRegex(OSError, "write"):
                module.extract_frames("video.webm", output_dir=output_directory)

    def test_image_read_failure(self):
        imagehash = types.ModuleType("imagehash")
        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(fromarray=lambda value: value)
        module = load_module(
            "frame_selector.py",
            {"cv2": fake_cv2(imread=None), "imagehash": imagehash, "PIL": pil},
        )
        with self.assertRaisesRegex(OSError, "read frame"):
            module._crop_to_detection("missing.jpg", {"corners": [[0, 0], [1, 1]]})

    def test_evidence_write_failure(self):
        module = load_module("evidence_manager.py", {"cv2": fake_cv2(imwrite=False)})
        with tempfile.TemporaryDirectory() as output_directory:
            with self.assertRaisesRegex(OSError, "write visual evidence"):
                module.save_evidence("m", "00_01", "slide", object(), {"text": "x", "average_confidence": .5}, output_directory)

    def test_empty_visual_evidence_is_valid(self):
        dependencies = {}
        for name, attributes in {
            "video_processor": {"extract_frames": lambda *a, **k: []},
            "frame_detector": {"detect_frames_with_visual_content": lambda frames: [], "classify_region_type": lambda x: "unknown"},
            "frame_selector": {"filter_unique_frames": lambda frames: []},
            "image_preprocessor": {"correct_perspective": lambda *a: None},
            "ocr_processor": {"run_ocr": lambda image: {}},
            "evidence_manager": {"save_evidence": lambda **kwargs: {}},
        }.items():
            dependency = types.ModuleType(name)
            for key, value in attributes.items():
                setattr(dependency, key, value)
            dependencies[name] = dependency
        module = load_module("pipeline.py", dependencies)
        self.assertEqual(
            module.process_meeting_video("video.webm", "meeting"),
            {"meeting_id": "meeting", "visual_evidence": []},
        )
