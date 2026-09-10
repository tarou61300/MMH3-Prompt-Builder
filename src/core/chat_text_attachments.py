from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from pathlib import Path


MAX_TEXT_ATTACHMENT_BYTES = 512 * 1024
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".csv",
    ".tsv", ".xml", ".html", ".htm", ".css", ".js", ".jsx", ".ts",
    ".tsx", ".py", ".pyw", ".rpy", ".sh", ".bash", ".zsh", ".ps1",
    ".bat", ".cmd", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".java", ".kt", ".kts", ".cs", ".go", ".rs", ".rb", ".php",
    ".swift", ".sql", ".srt", ".vtt",
}


class ChatTextFileError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _decode_strict(data: bytes) -> str:
    try:
        if data.startswith(codecs.BOM_UTF8):
            return data.decode("utf-8-sig")
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return data.decode("utf-16")
        if b"\x00" in data:
            raise ChatTextFileError("CHAT_TEXT_FILE_BINARY")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("cp932")
    except ChatTextFileError:
        raise
    except UnicodeDecodeError as exc:
        raise ChatTextFileError("CHAT_TEXT_FILE_DECODE_FAILED") from exc


def _has_excessive_control_characters(text: str) -> bool:
    invalid = sum(
        1
        for character in text
        if (ord(character) < 32 and character not in "\t\n\r")
        or 0x7F <= ord(character) <= 0x9F
    )
    return invalid >= 4 and invalid * 100 > max(1, len(text))


@dataclass(frozen=True, slots=True)
class ChatTextAttachment:
    """One in-memory text attachment; content and paths stay out of repr/logs."""

    filename: str
    size_bytes: int
    text: str = field(repr=False)
    source_path: str = field(default="", repr=False)

    @classmethod
    def from_file(cls, path: Path | str) -> "ChatTextAttachment":
        text_path = Path(path)
        if text_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
            raise ChatTextFileError("CHAT_TEXT_FILE_UNSUPPORTED_FORMAT")
        try:
            size_bytes = text_path.stat().st_size
        except OSError as exc:
            raise ChatTextFileError("CHAT_TEXT_FILE_READ_FAILED") from exc
        if size_bytes > MAX_TEXT_ATTACHMENT_BYTES:
            raise ChatTextFileError("CHAT_TEXT_FILE_TOO_LARGE")
        try:
            data = text_path.read_bytes()
        except OSError as exc:
            raise ChatTextFileError("CHAT_TEXT_FILE_READ_FAILED") from exc
        if len(data) > MAX_TEXT_ATTACHMENT_BYTES:
            raise ChatTextFileError("CHAT_TEXT_FILE_TOO_LARGE")
        text = _decode_strict(data)
        if _has_excessive_control_characters(text):
            raise ChatTextFileError("CHAT_TEXT_FILE_BINARY")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise ChatTextFileError("CHAT_TEXT_FILE_EMPTY")
        return cls(
            filename=text_path.name,
            size_bytes=len(data),
            text=text,
            source_path=str(text_path.resolve(strict=False)),
        )
