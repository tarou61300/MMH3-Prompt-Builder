from __future__ import annotations

import codecs
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from app.chat_page import ChatPage
from app.main_window import MainWindow
from app.workers import ChatThread
from core.chat_attachments import ChatImageAttachment
from core.chat_engine import ChatEngine
from core.chat_text_attachments import (
    MAX_TEXT_ATTACHMENT_BYTES,
    SUPPORTED_TEXT_EXTENSIONS,
    ChatTextAttachment,
    ChatTextFileError,
)
from core.config_manager import AppConfig, ConfigManager
from core.llama_manager import LlamaContextError
from core.localization import Localization, SUPPORTED_LOCALES
from mock_server import start_mock_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "skills" / "h3-prompt-writing"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(app: QApplication, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert predicate()


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ("plain UTF-8".encode(), "plain UTF-8"),
        (codecs.BOM_UTF8 + "BOM UTF-8".encode(), "BOM UTF-8"),
        (codecs.BOM_UTF16_LE + "UTF-16 LE".encode("utf-16-le"), "UTF-16 LE"),
        (codecs.BOM_UTF16_BE + "UTF-16 BE".encode("utf-16-be"), "UTF-16 BE"),
        ("日本語CP932".encode("cp932"), "日本語CP932"),
    ),
)
def test_text_attachment_decodes_supported_encodings(tmp_path, payload, expected):
    path = tmp_path / "sample.txt"
    path.write_bytes(payload)

    attachment = ChatTextAttachment.from_file(path)

    assert attachment.text == expected
    assert attachment.size_bytes == len(payload)


@pytest.mark.parametrize("suffix", (".txt", ".md", ".py", ".rpy", ".srt"))
def test_representative_text_extensions_are_supported(tmp_path, suffix):
    path = tmp_path / f"sample{suffix}"
    path.write_text("content", encoding="utf-8")

    assert suffix in SUPPORTED_TEXT_EXTENSIONS
    assert ChatTextAttachment.from_file(path).text == "content"


def test_text_attachment_normalizes_newlines_without_modifying_source_or_repr(tmp_path):
    path = tmp_path / "private source.txt"
    original = b"first\r\nsecond\rthird"
    path.write_bytes(original)

    attachment = ChatTextAttachment.from_file(path)
    rendered = repr(attachment)

    assert attachment.text == "first\nsecond\nthird"
    assert path.read_bytes() == original
    assert attachment.filename in rendered
    assert attachment.text not in rendered
    assert attachment.source_path not in rendered
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    ("name", "payload", "code"),
    (
        ("unsupported.pdf", b"text", "CHAT_TEXT_FILE_UNSUPPORTED_FORMAT"),
        ("broken.txt", b"\x81", "CHAT_TEXT_FILE_DECODE_FAILED"),
        ("nul.txt", b"plain\x00text", "CHAT_TEXT_FILE_BINARY"),
        ("controls.txt", b"ok\x01\x02\x03\x04", "CHAT_TEXT_FILE_BINARY"),
        ("empty.txt", b" \r\n\t ", "CHAT_TEXT_FILE_EMPTY"),
    ),
)
def test_text_attachment_rejects_unsafe_or_empty_content(
    tmp_path, name, payload, code
):
    path = tmp_path / name
    path.write_bytes(payload)

    with pytest.raises(ChatTextFileError, match=code):
        ChatTextAttachment.from_file(path)


def test_text_attachment_reports_read_failure_for_missing_supported_file(tmp_path):
    with pytest.raises(ChatTextFileError, match="CHAT_TEXT_FILE_READ_FAILED"):
        ChatTextAttachment.from_file(tmp_path / "missing.txt")


@pytest.mark.parametrize(
    ("size", "accepted"),
    ((MAX_TEXT_ATTACHMENT_BYTES, True), (MAX_TEXT_ATTACHMENT_BYTES + 1, False)),
)
def test_text_attachment_enforces_exact_512_kib_limit(tmp_path, size, accepted):
    path = tmp_path / "limit.txt"
    path.write_bytes(b"a" * size)

    if accepted:
        assert ChatTextAttachment.from_file(path).size_bytes == size
    else:
        with pytest.raises(ChatTextFileError, match="CHAT_TEXT_FILE_TOO_LARGE"):
            ChatTextAttachment.from_file(path)


def test_chat_engine_wraps_text_file_as_string_with_question_and_no_path(tmp_path):
    path = tmp_path / "private notes.md"
    path.write_text("alpha\nbeta", encoding="utf-8")
    attachment = ChatTextAttachment.from_file(path)

    payload = ChatEngine().request_payload(
        [{"role": "user", "content": "What does alpha mean?", "text_file": attachment}]
    )
    content = payload["messages"][-1]["content"]

    assert isinstance(content, str)
    assert content == (
        "Attached local text file: private notes.md\n"
        "--- BEGIN ATTACHED FILE ---\n"
        "alpha\nbeta\n"
        "--- END ATTACHED FILE ---\n\n"
        "User request:\n"
        "What does alpha mean?"
    )
    assert attachment.source_path not in content


def test_chat_engine_uses_localized_file_only_instruction_and_retains_context(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("remember this value: 42", encoding="utf-8")
    attachment = ChatTextAttachment.from_file(path)
    conversation = [
        {"role": "user", "content": "", "text_file": attachment},
        {"role": "assistant", "content": "The value is 42."},
        {"role": "user", "content": "What was the value?"},
    ]

    payload = ChatEngine(
        text_file_only_instruction="このファイルの内容を要約してください。"
    ).request_payload(conversation)

    first_user = payload["messages"][1]["content"]
    assert "remember this value: 42" in first_user
    assert "このファイルの内容を要約してください。" in first_user
    assert payload["messages"][-1]["content"] == "What was the value?"
    assert conversation[0]["content"] == ""


def test_chat_engine_rejects_image_and_text_file_in_the_same_turn(tmp_path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("text", encoding="utf-8")
    text_attachment = ChatTextAttachment.from_file(text_path)
    image_attachment = ChatImageAttachment(
        filename="image.png",
        mime_type="image/png",
        _data=b"\x89PNG\r\n\x1a\n",
    )

    with pytest.raises(ValueError, match="CHAT_CONVERSATION_INVALID"):
        ChatEngine().request_payload(
            [
                {
                    "role": "user",
                    "content": "inspect",
                    "image": image_attachment,
                    "text_file": text_attachment,
                }
            ]
        )


def test_existing_image_payload_shape_is_unchanged():
    image = ChatImageAttachment(
        filename="image.png",
        mime_type="image/png",
        _data=b"\x89PNG\r\n\x1a\n",
    )

    content = ChatEngine().request_payload(
        [{"role": "user", "content": "Describe.", "image": image}]
    )["messages"][-1]["content"]

    assert content == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
        },
        {"type": "text", "text": "Describe."},
    ]


def test_text_attachment_controls_and_file_only_send(tmp_path):
    app = _app()
    page = ChatPage(Localization(PROJECT_ROOT / "locales", "ja-JP").tr)
    path = tmp_path / "sample.py"
    path.write_text("print('hello')", encoding="utf-8")
    attachment = ChatTextAttachment.from_file(path)
    sent = []
    page.send_requested.connect(lambda text, item: sent.append((text, item)))
    page.show()
    try:
        page.set_attachment(attachment)
        app.processEvents()

        assert page.text_file_button.isVisibleTo(page)
        assert page.attachment is attachment
        assert page.attachment_label.text() == "sample.py"
        assert page.attachment_thumbnail.text() == "TXT"
        assert page.analyze_button.isHidden()
        assert page.reference_analyze_button.isHidden()
        assert page.send_button.isEnabled()
        page.send_button.click()
        assert sent == [("", attachment)]
        page.remove_attachment_button.click()
        assert page.attachment is None
        assert not page.send_button.isEnabled()
    finally:
        page.close()


def test_text_file_picker_does_not_require_mmproj(tmp_path, monkeypatch):
    app = _app()
    path = tmp_path / "notes.txt"
    path.write_text("content", encoding="utf-8")
    manager = ConfigManager(tmp_path / "data")
    manager.save(AppConfig(model_path=str(tmp_path / "model.gguf")))
    window = MainWindow(
        project_root=PROJECT_ROOT,
        config_manager=manager,
        server_url="http://127.0.0.1:54321",
        dev_skill_path=SKILL_FIXTURE,
    )
    monkeypatch.setattr(
        "app.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(path), "Text files"),
    )
    try:
        window.chat_page.text_file_button.click()
        app.processEvents()
        assert isinstance(window.chat_page.attachment, ChatTextAttachment)
        assert window.chat_page.attachment.filename == "notes.txt"
        assert window.chat_page.mmproj_guidance.isHidden()
    finally:
        window.close()
        app.processEvents()


def test_drag_and_drop_attaches_supported_text_file(tmp_path):
    app = _app()
    page = ChatPage(Localization(PROJECT_ROOT / "locales", "en-US").tr)
    path = tmp_path / "drop.md"
    path.write_text("# Heading", encoding="utf-8")
    dropped = []
    page.text_file_path_requested.connect(dropped.append)
    page.show()
    try:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        enter = QDragEnterEvent(
            QPoint(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(page, enter)
        assert enter.isAccepted()
        drop = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(page, drop)
        assert drop.isAccepted()
        assert len(dropped) == 1
        assert Path(dropped[0]) == path
    finally:
        page.close()


def test_text_attachment_busy_state_and_new_chat_clear(tmp_path):
    app = _app()
    path = tmp_path / "notes.txt"
    path.write_text("content", encoding="utf-8")
    manager = ConfigManager(tmp_path / "data")
    manager.save(AppConfig())
    window = MainWindow(
        project_root=PROJECT_ROOT,
        config_manager=manager,
        server_url="http://127.0.0.1:54321",
        dev_skill_path=SKILL_FIXTURE,
    )
    try:
        window.chat_page.set_attachment(ChatTextAttachment.from_file(path))
        window.chat_page.set_busy(any_llm_busy=True, chat_busy=True)
        assert not window.chat_page.text_file_button.isEnabled()
        assert not window.chat_page.remove_attachment_button.isEnabled()
        window.chat_page.set_busy(any_llm_busy=False, chat_busy=False)
        assert window.chat_page.text_file_button.isEnabled()
        assert window.chat_page.remove_attachment_button.isEnabled()
        window._new_chat()
        assert window.chat_page.attachment is None
    finally:
        window.close()
        app.processEvents()


def test_text_file_and_message_send_success_keeps_context_without_persistence(
    tmp_path, caplog
):
    app = _app()
    secret = "private attachment sentinel 4711"
    path = tmp_path / "private source.txt"
    path.write_text(secret, encoding="utf-8")
    data_dir = tmp_path / "portable-data"
    manager = ConfigManager(data_dir)
    manager.save(AppConfig())
    mock, url = start_mock_server(response_text="The file contains a sentinel.")
    window = MainWindow(
        project_root=PROJECT_ROOT,
        config_manager=manager,
        server_url=url,
        dev_skill_path=SKILL_FIXTURE,
    )
    caplog.set_level(logging.DEBUG)
    try:
        window._attach_chat_text_file(str(path))
        attachment = window.chat_page.attachment
        window.chat_page.input_text.setPlainText("Explain it.")
        window.chat_page.send_button.click()
        _wait_until(app, lambda: not window._chat_active)

        content = mock.received_payloads[-1]["messages"][-1]["content"]
        assert secret in content
        assert "Explain it." in content
        assert str(path.resolve()) not in content
        assert window.chat_page.attachment is None
        assert window.chat_messages[0]["text_file"] is attachment

        window.chat_page.input_text.setPlainText("What was attached?")
        window.chat_page.send_button.click()
        _wait_until(app, lambda: not window._chat_active)
        assert secret in mock.received_payloads[-1]["messages"][1]["content"]
        assert secret not in caplog.text
        assert str(path.resolve()) not in caplog.text
        for stored in data_dir.rglob("*"):
            if stored.is_file():
                assert secret.encode() not in stored.read_bytes()
    finally:
        window.close()
        app.processEvents()
        mock.shutdown()
        mock.server_close()


def test_failed_text_file_turn_rolls_back_and_restores_draft_and_attachment(tmp_path):
    app = _app()
    path = tmp_path / "failed.txt"
    path.write_text("content", encoding="utf-8")
    attachment = ChatTextAttachment.from_file(path)
    manager = ConfigManager(tmp_path / "data")
    manager.save(AppConfig())
    window = MainWindow(
        project_root=PROJECT_ROOT,
        config_manager=manager,
        server_url="http://127.0.0.1:54321",
        dev_skill_path=SKILL_FIXTURE,
    )
    try:
        window.chat_page.set_attachment(attachment)
        pending = {
            "role": "user",
            "content": "Summarize it.",
            "text_file": attachment,
        }
        window.chat_messages.append(pending)
        window._pending_chat_user_message = pending
        window._pending_chat_draft = "Summarize it."
        window._pending_chat_message_widget = window.chat_page.add_message(
            "user",
            "Summarize it.",
            text_filename=attachment.filename,
        )

        window._chat_error("CHAT_CONTEXT_OVERFLOW")
        app.processEvents()

        assert window.chat_messages == []
        assert window.chat_page.attachment is attachment
        assert window.chat_page.input_text.toPlainText() == "Summarize it."
        assert window.tr("chat.error.context") in window.chat_page.status_label.text()
    finally:
        window.close()
        app.processEvents()


class _TextFakeServer:
    def __init__(self, *, overflow: bool = False):
        self.starts = []
        self.payloads = []
        self.preflight_calls = 0
        self.generate_calls = 0
        self.overflow = overflow

    def start(self, model_path, **settings):
        self.starts.append((str(model_path), settings))

    def preflight_context(self, payload, context_size):
        self.preflight_calls += 1
        self.payloads.append(payload)
        if self.overflow:
            raise LlamaContextError("too large")
        return 100, 512

    def generate(self, payload, timeout):
        self.generate_calls += 1
        return "text answer"


@pytest.mark.parametrize(
    ("overflow", "expected_result", "expected_error"),
    ((False, "text answer", None), (True, None, "CHAT_CONTEXT_OVERFLOW")),
)
def test_text_worker_needs_no_mmproj_and_always_runs_context_preflight(
    tmp_path, monkeypatch, overflow, expected_result, expected_error
):
    path = tmp_path / "notes.txt"
    path.write_text("content", encoding="utf-8")
    model = tmp_path / "model.gguf"
    config = AppConfig(model_path=str(model))
    monkeypatch.setattr(
        "app.workers.validate_model", lambda value: SimpleNamespace(path=Path(value))
    )
    server = _TextFakeServer(overflow=overflow)
    thread = ChatThread(
        engine=ChatEngine(),
        server=server,
        config=config,
        conversation=[
            {
                "role": "user",
                "content": "Summarize.",
                "text_file": ChatTextAttachment.from_file(path),
            }
        ],
        mock_mode=False,
    )
    results = []
    errors = []
    thread.result_ready.connect(results.append)
    thread.error_occurred.connect(errors.append)

    thread.run()

    assert server.preflight_calls == 1
    assert "mmproj_path" not in server.starts[0][1]
    assert results == ([] if expected_result is None else [expected_result])
    assert errors == ([] if expected_error is None else [expected_error])
    assert server.generate_calls == (0 if overflow else 1)


def test_all_locales_have_text_attachment_messages_and_matching_keys():
    key_sets = []
    for locale_id in SUPPORTED_LOCALES:
        localization = Localization(PROJECT_ROOT / "locales", locale_id)
        locale_values = json.loads(
            (PROJECT_ROOT / "locales" / f"{locale_id}.json").read_text(
                encoding="utf-8"
            )
        )
        key_sets.append(set(locale_values))
        for key in (
            "chat.add_text_file",
            "chat.add_text_file_tooltip",
            "chat.remove_attachment",
            "chat.text_file_message",
            "chat.text_file.choose",
            "chat.text_file.filter",
            "chat.text_file_only_instruction",
            "chat.error.text_file_format",
            "chat.error.text_file_read",
            "chat.error.text_file_decode",
            "chat.error.text_file_binary",
            "chat.error.text_file_too_large",
            "chat.error.text_file_empty",
        ):
            assert localization.tr(key) != key
    assert all(keys == key_sets[0] for keys in key_sets[1:])
