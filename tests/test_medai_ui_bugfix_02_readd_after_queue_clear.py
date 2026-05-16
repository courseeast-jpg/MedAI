from __future__ import annotations

from pathlib import Path

from app.main import (
    PERSISTED_UPLOAD_FINGERPRINTS_KEY,
    PERSISTED_UPLOAD_GENERATION_KEY,
    UPLOAD_WIDGET_VERSION_KEY,
    clear_last_report_action,
    clear_queue_action,
    current_upload_widget_key,
    persist_uploaded_files_once,
    persisted_upload_fingerprints,
)


class FakeUpload:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content
        self.size = len(content)

    def getbuffer(self) -> memoryview:
        return memoryview(self._content)


def _three_uploads() -> list[FakeUpload]:
    return [
        FakeUpload("first.pdf", b"alpha"),
        FakeUpload("second.pdf", b"beta"),
        FakeUpload("third.txt", b"gamma"),
    ]


def test_readding_same_files_after_queue_clear_persists_new_batch() -> None:
    session_state: dict = {}
    queue: list[Path] = []

    def save_func(uploaded_file) -> Path:
        path = Path(f"queued_{len(queue)}_{uploaded_file.name}")
        queue.append(path)
        return path

    def clear_queue_func() -> list[Path]:
        removed = list(queue)
        queue.clear()
        return removed

    uploads = _three_uploads()
    first = persist_uploaded_files_once(uploads, session_state, save_func=save_func)
    second = persist_uploaded_files_once(uploads, session_state, save_func=save_func)

    assert len(first) == 3
    assert second == []
    assert len(queue) == 3

    removed = clear_queue_action(session_state, clear_func=clear_queue_func)

    assert len(removed) == 3
    assert queue == []
    assert persisted_upload_fingerprints(session_state) == set()

    readded = persist_uploaded_files_once(uploads, session_state, save_func=save_func)

    assert len(readded) == 3
    assert len(queue) == 3


def test_clear_last_report_does_not_change_queue_count_or_duplicate_uploads() -> None:
    session_state: dict = {}
    queue: list[Path] = []

    def save_func(uploaded_file) -> Path:
        path = Path(f"queued_{len(queue)}_{uploaded_file.name}")
        queue.append(path)
        return path

    uploads = _three_uploads()
    persist_uploaded_files_once(uploads, session_state, save_func=save_func)
    before_fingerprints = set(session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY])
    before_generation = session_state[PERSISTED_UPLOAD_GENERATION_KEY]
    before_widget_version = session_state.get(UPLOAD_WIDGET_VERSION_KEY, 0)

    removed = clear_last_report_action(session_state, clear_func=lambda: [Path("latest_test_run.md")])
    rerun_saved = persist_uploaded_files_once(uploads, session_state, save_func=save_func)
    clear_last_report_action(session_state, clear_func=lambda: [])
    second_rerun_saved = persist_uploaded_files_once(uploads, session_state, save_func=save_func)

    assert removed == [Path("latest_test_run.md")]
    assert rerun_saved == []
    assert second_rerun_saved == []
    assert len(queue) == 3
    assert session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY] == before_fingerprints
    assert session_state[PERSISTED_UPLOAD_GENERATION_KEY] == before_generation
    assert session_state.get(UPLOAD_WIDGET_VERSION_KEY, 0) == before_widget_version


def test_remove_queued_files_resets_upload_tracking_and_widget_key() -> None:
    session_state: dict = {}
    uploads = _three_uploads()
    persist_uploaded_files_once(uploads, session_state, save_func=lambda uploaded: Path(uploaded.name))
    before_key = current_upload_widget_key(session_state)

    clear_queue_action(session_state, clear_func=lambda: [Path("first.pdf"), Path("second.pdf"), Path("third.txt")])

    after_key = current_upload_widget_key(session_state)

    assert after_key != before_key
    assert session_state[UPLOAD_WIDGET_VERSION_KEY] == 1
    assert session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY] == set()
    assert session_state[PERSISTED_UPLOAD_GENERATION_KEY] == 1


def test_clear_last_report_does_not_reset_upload_tracking_or_widget_key() -> None:
    session_state: dict = {}
    uploads = _three_uploads()
    persist_uploaded_files_once(uploads, session_state, save_func=lambda uploaded: Path(uploaded.name))
    before_key = current_upload_widget_key(session_state)
    before_fingerprints = set(session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY])

    clear_last_report_action(session_state, clear_func=lambda: [])

    assert current_upload_widget_key(session_state) == before_key
    assert session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY] == before_fingerprints


def test_generation_mismatch_is_treated_as_clean_upload_session() -> None:
    session_state = {
        PERSISTED_UPLOAD_FINGERPRINTS_KEY: {"old_fingerprint"},
        PERSISTED_UPLOAD_GENERATION_KEY: 0,
        UPLOAD_WIDGET_VERSION_KEY: 1,
    }
    saved: list[Path] = []

    def save_func(uploaded_file) -> Path:
        path = Path(uploaded_file.name)
        saved.append(path)
        return path

    persisted = persist_uploaded_files_once([FakeUpload("same.pdf", b"alpha")], session_state, save_func=save_func)

    assert persisted == [Path("same.pdf")]
    assert saved == [Path("same.pdf")]
    assert "old_fingerprint" not in session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY]


def test_start_run_invocation_remains_unchanged() -> None:
    source = Path("app/main.py").read_text(encoding="utf-8")

    assert 'run_medai_test_batch(sys_components["execution"], specialty=specialty)' in source
