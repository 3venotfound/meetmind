import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


AI_PATH = Path(__file__).resolve().parents[2] / "ai" / "ai_functions.py"


class State:
    def __init__(self, name):
        self.name = name


class RemoteFile:
    def __init__(self, name, state):
        self.name = name
        self.state = State(state)


class FakeFiles:
    def __init__(self, states):
        self.states = list(states)
        self.deleted = []
        self.delete_error = False
        self.upload_config = None
        self.remote = RemoteFile("files/controlled", self.states.pop(0))

    def upload(self, *, file, config):
        self.upload_config = config
        return self.remote

    def get(self, *, name):
        self.remote = RemoteFile(name, self.states.pop(0))
        return self.remote

    def delete(self, *, name):
        self.deleted.append(name)
        if self.delete_error:
            raise RuntimeError("delete failed")


class FakeModels:
    def __init__(self, malformed=False):
        self.calls = 0
        self.malformed = malformed

    def generate_content(self, **kwargs):
        self.calls += 1
        payload = {
            "transcript": "Sarah: Budget is RM70,000.",
            "summary": "Budget confirmed.",
            "decisions": [], "action_items": [],
            "budget": {"value": "RM70,000", "mentioned_by": "Sarah", "timestamp_seconds": 1},
            "deadline": {"value": None, "mentioned_by": None},
            "owner": {"value": None, "mentioned_by": None},
        }
        return types.SimpleNamespace(text="not-json" if self.malformed else json.dumps(payload))


class FakeClient:
    def __init__(self, states, malformed=False):
        self.files = FakeFiles(states)
        self.models = FakeModels(malformed)


def load_ai(fake_client):
    fake_types = types.SimpleNamespace(
        UploadFileConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = lambda **kwargs: fake_client
    genai_module.types = fake_types
    google_module = types.ModuleType("google")
    google_module.genai = genai_module
    name = f"meetmind_ai_test_{id(fake_client)}"
    spec = importlib.util.spec_from_file_location(name, AI_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}):
        spec.loader.exec_module(module)
    return module


class GeminiFileLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        temporary.write(b"video")
        temporary.close()
        self.path = Path(temporary.name)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def call(self, states, **kwargs):
        client = FakeClient(states, kwargs.pop("malformed", False))
        module = load_ai(client)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=False):
            result = module.extract_from_audio(
                str(self.path), poll_interval_seconds=0.001,
                file_timeout_seconds=1, **kwargs,
            )
        return client, result

    def test_immediate_active_and_remote_deletion(self):
        client = FakeClient(["ACTIVE"])
        module = load_ai(client)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=False):
            result = module.extract_from_audio(
                str(self.path), poll_interval_seconds=0.001,
                file_timeout_seconds=1,
            )
        self.assertEqual(result["summary"], "Budget confirmed.")
        self.assertEqual(client.files.deleted, ["files/controlled"])
        self.assertEqual(client.files.upload_config.mime_type, "video/webm")
        self.assertEqual(module._last_remote_deletion_state, "succeeded")

    def test_processing_then_active(self):
        client, _ = self.call(["PROCESSING", "ACTIVE"])
        self.assertEqual(client.models.calls, 1)
        self.assertEqual(client.files.deleted, ["files/controlled"])

    def test_failed_state_deletes_remote_file(self):
        client = FakeClient(["FAILED"])
        module = load_ai(client)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=False):
            with self.assertRaises(RuntimeError) as context:
                module.extract_from_audio(str(self.path), file_timeout_seconds=1, poll_interval_seconds=.001)
        self.assertEqual(client.files.deleted, ["files/controlled"])
        self.assertEqual(context.exception._meetmind_deletion_state, "succeeded")
        self.assertEqual(module._last_remote_deletion_state, "succeeded")

    def test_timeout_deletes_remote_file(self):
        client = FakeClient(["PROCESSING"])
        module = load_ai(client)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=False), patch.object(
            module.time, "monotonic", side_effect=[0.0, 2.0]
        ):
            with self.assertRaises(TimeoutError):
                module.extract_from_audio(str(self.path), file_timeout_seconds=1, poll_interval_seconds=.001)
        self.assertEqual(client.files.deleted, ["files/controlled"])

    def test_extraction_failure_still_deletes_remote_file(self):
        client = FakeClient(["ACTIVE"], malformed=True)
        module = load_ai(client)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=False):
            with self.assertRaises(ValueError):
                module.extract_from_audio(str(self.path), max_retries=0)
        self.assertEqual(client.files.deleted, ["files/controlled"])

    def test_processing_error_is_preserved_when_deletion_also_fails(self):
        client = FakeClient(["FAILED"])
        client.files.delete_error = True
        module = load_ai(client)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "processing failed") as context:
                module.extract_from_audio(str(self.path))
        self.assertEqual(context.exception._meetmind_deletion_state, "failed")
        self.assertEqual(module._last_remote_deletion_state, "failed")

    def test_remote_deletion_failure_after_success_is_reported_safely(self):
        client = FakeClient(["ACTIVE"])
        client.files.delete_error = True
        module = load_ai(client)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "delete failed") as context:
                module.extract_from_audio(str(self.path))
        self.assertEqual(context.exception._meetmind_diagnostic["stage"], "deletion")
        self.assertEqual(context.exception._meetmind_deletion_state, "failed")
        self.assertEqual(module._last_remote_deletion_state, "failed")

    def test_missing_api_key(self):
        client = FakeClient(["ACTIVE"])
        module = load_ai(client)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "not configured"):
                module.extract_from_audio(str(self.path))
        self.assertEqual(client.files.deleted, [])

    def test_safe_diagnostic_never_copies_exception_text(self):
        client = FakeClient(["ACTIVE"])
        module = load_ai(client)

        class ProviderError(RuntimeError):
            code = 403
            status = "PERMISSION_DENIED"
            message = (
                "token=secret-test-key at C:\\private\\recording.mp4 "
                "see https://provider.example/error?key=secret-test-key "
                + "x" * 400
            )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-test-key"}, clear=False):
            diagnostic = module._safe_diagnostic(ProviderError(), "generation")

        self.assertEqual(diagnostic["stage"], "generation")
        self.assertEqual(diagnostic["exception_class"], "ProviderError")
        self.assertEqual(diagnostic["status_code"], 403)
        self.assertEqual(diagnostic["provider_category"], "PERMISSION_DENIED")
        self.assertEqual(diagnostic["message"], "AI generation failed.")
        self.assertLessEqual(len(diagnostic["message"]), 240)
        self.assertNotIn("secret-test-key", diagnostic["message"])
        self.assertNotIn("recording.mp4", diagnostic["message"])
        self.assertNotIn("provider.example", diagnostic["message"])


class AIValidationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_ai(FakeClient(["ACTIVE"]))

    @staticmethod
    def extraction(**overrides):
        payload = {
            "decisions": [
                {
                    "text": "Proceed with launch.",
                    "decided_by": "Sarah",
                    "timestamp_seconds": None,
                }
            ],
            "action_items": [],
            "budget": {"value": None, "mentioned_by": "Sarah"},
            "deadline": {"value": None, "mentioned_by": None},
            "owner": {"value": None, "mentioned_by": None},
        }
        payload.update(overrides)
        return payload

    def test_non_speaking_action_item_owner_is_accepted_and_preserved(self):
        data = self.extraction(
            action_items=[
                {"description": "Prepare the launch checklist.", "owner": "Amina"}
            ]
        )

        issues = self.module._validate(data, {"Sarah"})

        self.assertEqual(issues, [])
        self.assertEqual(data["action_items"][0]["owner"], "Amina")

    def test_empty_whitespace_and_untrimmed_action_item_owners_are_rejected(self):
        for owner in ("", "   ", " Amina", "Amina "):
            with self.subTest(owner=repr(owner)):
                data = self.extraction(
                    action_items=[
                        {"description": "Prepare the checklist.", "owner": owner}
                    ]
                )
                issues = self.module._validate(data, {"Sarah"})
                self.assertIn(
                    "action_item.owner must be a non-empty trimmed string",
                    issues,
                )
                self.assertEqual(data["action_items"][0]["owner"], owner)

    def test_invalid_decided_by_and_mentioned_by_remain_rejected(self):
        data = self.extraction(
            decisions=[
                {
                    "text": "Proceed with launch.",
                    "decided_by": "Unknown Person",
                    "timestamp_seconds": None,
                }
            ],
            budget={"value": "RM70,000", "mentioned_by": "Unknown Person"},
            deadline={"value": "15 August", "mentioned_by": "Unknown Person"},
            owner={"value": "Amina", "mentioned_by": "Unknown Person"},
        )

        issues = self.module._validate(data, {"Sarah"})

        self.assertEqual(
            issues,
            [
                "decision.decided_by is not a speaker in this transcript",
                "budget.mentioned_by is not a speaker in this transcript",
                "deadline.mentioned_by is not a speaker in this transcript",
                "owner.mentioned_by is not a speaker in this transcript",
            ],
        )
