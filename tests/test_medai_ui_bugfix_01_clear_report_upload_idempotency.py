from __future__ import annotations

from pathlib import Path

from app.main import (
    PERSISTED_UPLOAD_FINGERPRINTS_KEY,
    UPLOAD_WIDGET_VERSION_KEY,
    clear_last_report_action,
    clear_queue_action,
    persist_uploaded_files_once,
    uploaded_file_fingerprint,
)


class FakeUpload:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content
        self.size = len(content)

    def getbuffer(self) -> memoryview:
        return memoryview(self._content)


def test_same_uploaded_file_batch_is_persisted_once() -> None:
    session_state: dict = {}
    upload = FakeUpload("sample.pdf", b"alpha")
    saved_paths: list[Path] = []

    def save_func(uploaded_file) -> Path:
        path = Path(f"saved_{len(saved_paths)}_{uploaded_file.name}")
        saved_paths.append(path)
        return path

    first = persist_uploaded_files_once([upload], session_state, save_func=save_func)
    second = persist_uploaded_files_once([upload], session_state, save_func=save_func)

    assert len(first) == 1
    assert second == []
    assert len(saved_paths) == 1
    assert uploaded_file_fingerprint(upload) in session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY]


def test_clear_last_report_does_not_change_queue_or_upload_tracking() -> None:
    session_state = {
        PERSISTED_UPLOAD_FINGERPRINTS_KEY: {"already_saved"},
        UPLOAD_WIDGET_VERSION_KEY: 3,
        "phase52_current_run": {"run_id": "run_1"},
    }
    queue_count = 2
    clear_calls = {"count": 0}

    def clear_func() -> list[Path]:
        clear_calls["count"] += 1
        return [Path("latest_test_run.md")]

    removed = clear_last_report_action(session_state, clear_func=clear_func)

    assert removed == [Path("latest_test_run.md")]
    assert queue_count == 2
    assert clear_calls["count"] == 1
    assert session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY] == {"already_saved"}
    assert session_state[UPLOAD_WIDGET_VERSION_KEY] == 3
    assert "phase52_current_run" not in session_state


def test_clear_last_report_does_not_call_upload_persistence() -> None:
    session_state: dict = {}
    persistence_called = {"value": False}

    def clear_func() -> list[Path]:
        return []

    clear_last_report_action(session_state, clear_func=clear_func)

    assert persistence_called["value"] is False


def test_remove_queued_files_clears_queue_and_resets_upload_tracking() -> None:
    session_state = {
        PERSISTED_UPLOAD_FINGERPRINTS_KEY: {"already_saved"},
        UPLOAD_WIDGET_VERSION_KEY: 4,
        "phase52_current_run": {"run_id": "run_1"},
    }

    def clear_func() -> list[Path]:
        return [Path("queued.pdf")]

    removed = clear_queue_action(session_state, clear_func=clear_func)

    assert removed == [Path("queued.pdf")]
    assert session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY] == set()
    assert session_state[UPLOAD_WIDGET_VERSION_KEY] == 5
    assert "phase52_current_run" not in session_state


def test_new_file_content_with_same_name_can_still_be_added() -> None:
    session_state: dict = {}
    first = FakeUpload("sample.pdf", b"alpha")
    second = FakeUpload("sample.pdf", b"beta")
    saved_paths: list[Path] = []

    def save_func(uploaded_file) -> Path:
        path = Path(f"saved_{len(saved_paths)}_{uploaded_file.name}")
        saved_paths.append(path)
        return path

    persist_uploaded_files_once([first], session_state, save_func=save_func)
    saved_second = persist_uploaded_files_once([second], session_state, save_func=save_func)

    assert len(saved_second) == 1
    assert len(saved_paths) == 2


def test_distinct_uploads_with_same_name_keep_duplicate_filename_path_available() -> None:
    session_state: dict = {}
    uploads = [FakeUpload("duplicate.pdf", b"alpha"), FakeUpload("duplicate.pdf", b"beta")]
    saved_names: list[str] = []

    def save_func(uploaded_file) -> Path:
        suffix = "" if not saved_names else "_1"
        path = Path(f"duplicate{suffix}.pdf")
        saved_names.append(path.name)
        return path

    saved = persist_uploaded_files_once(uploads, session_state, save_func=save_func)

    assert [path.name for path in saved] == ["duplicate.pdf", "duplicate_1.pdf"]
    assert len(saved_names) == 2
