"""Safe Unicode/HTML normalization and bounded decoding for scanner visibility."""
from __future__ import annotations

import base64
import codecs
import html
import re
import unicodedata
from dataclasses import dataclass

ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
HIDDEN_HTML = re.compile(r"<!--(.*?)-->", re.DOTALL)
HIDDEN_STYLE = re.compile(
    r"<(span|div|p|small|font)\b([^>]*(?:display\s*:\s*none|visibility\s*:\s*hidden|"
    r"font-size\s*:\s*(?:0|1px)|opacity\s*:\s*0|color\s*:\s*#?fff(?:fff)?\s*;?\s*background(?:-color)?\s*:\s*#?fff(?:fff)?)[^>]*)>(.*?)</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
B64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{12,}={0,2}(?![A-Za-z0-9+/])")
HEX_TOKEN = re.compile(r"\b(?:[0-9a-fA-F]{2}){4,}\b")
CONFUSABLES = str.maketrans({
    "а": "a", "А": "A", "е": "e", "Е": "E", "о": "o", "О": "O", "р": "p", "Р": "P",
    "с": "c", "С": "C", "х": "x", "Х": "X", "у": "y", "У": "Y", "і": "i", "І": "I",
    "ј": "j", "Ј": "J", "ԁ": "d", "ԛ": "q", "ԝ": "w", "Ь": "b",
})


@dataclass
class NormalizationResult:
    """Normalized text and security signals discovered during safe transformations."""
    text: str
    signals: list[str]
    removed_count: int = 0


def _printable_decoded(value: bytes, limit: int = 2000) -> str:
    """Decode bounded bytes only when UTF-8 yields mostly printable content."""
    try:
        text = value[:limit].decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return ""
    if not text:
        return ""
    printable = sum(char.isprintable() or char in "\r\n\t" for char in text) / len(text)
    return text if printable >= 0.85 else ""


def normalize_content(content: str, decode_limit: int = 20) -> NormalizationResult:
    """Normalize text, surface hidden/encoded content, and report each transformation.

    This function never executes input. Decoding is capped by candidate count and byte size.
    """
    original = str(content)
    signals: list[str] = []
    removed_count = 0
    text = unicodedata.normalize("NFKC", original)
    if text != original:
        signals.append("unicode_nfkc_changed")
    cleaned = ZERO_WIDTH.sub("", text)
    if cleaned != text:
        removed_count = len(text) - len(cleaned)
        signals.append("zero_width_or_bidi_removed")
    text = cleaned.translate(CONFUSABLES)
    if text != cleaned:
        signals.append("confusable_folded")

    def expose_comment(match: re.Match[str]) -> str:
        signals.append("hidden_html_comment")
        return "\n hidden-comment-content: " + html.unescape(match.group(1)) + " \n"
    text = HIDDEN_HTML.sub(expose_comment, text)

    def expose_hidden(match: re.Match[str]) -> str:
        signals.append("hidden_html_style")
        return "\n hidden-style-content: " + html.unescape(match.group(3)) + " \n"
    text = HIDDEN_STYLE.sub(expose_hidden, text)
    text = html.unescape(text)

    candidates = 0
    for pattern, decoder, signal in (
        (B64_TOKEN, lambda token: base64.b64decode(token, validate=True), "base64_candidate"),
        (HEX_TOKEN, bytes.fromhex, "hex_candidate"),
    ):
        for match in list(pattern.finditer(text)):
            if candidates >= max(0, decode_limit):
                break
            token = match.group(0)
            try:
                decoded = _printable_decoded(decoder(token))
            except (ValueError, TypeError, base64.binascii.Error):
                decoded = ""
            if decoded:
                signals.append(signal)
                text += f"\n decoded-{signal}: {decoded}"
                candidates += 1
    for match in re.finditer(r"(?i)\brot13\s*:\s*([a-zA-Z0-9 .,!?_-]{8,160})", text):
        if candidates >= max(0, decode_limit):
            break
        decoded = codecs.decode(match.group(1), "rot_13")
        if decoded and sum(ch.isprintable() for ch in decoded) / len(decoded) > 0.9:
            signals.append("rot13_candidate")
            text += f"\n decoded-rot13: {decoded}"
            candidates += 1
    return NormalizationResult(text=text, signals=list(dict.fromkeys(signals)), removed_count=removed_count)
