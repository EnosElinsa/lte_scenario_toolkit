"""Stable English and Simplified Chinese interface translations."""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Final

DEFAULT_LANGUAGE: Final = "en"
SUPPORTED_LANGUAGES: Final = ("en", "zh-CN")

TRANSLATIONS: Final[dict[str, dict[str, str]]] = {
    "en": {
        "app.title": "LTE Scenario Toolkit",
        "nav.scenarios": "Scenarios",
        "nav.configure": "Configure",
        "nav.figures": "Figures",
        "nav.history": "History",
        "status.ready": "Ready",
        "status.idle": "No active job",
        "action.validate": "Validate",
        "error.job_busy": "Another job is already running",
        "job.running": "Running {name}",
        "label.language": "Language",
    },
    "zh-CN": {
        "app.title": "LTE \u573a\u666f\u5de5\u5177\u7bb1",
        "nav.scenarios": "\u573a\u666f",
        "nav.configure": "\u914d\u7f6e",
        "nav.figures": "\u56fe\u8868",
        "nav.history": "\u5386\u53f2",
        "status.ready": "\u5c31\u7eea",
        "status.idle": "\u65e0\u6d3b\u52a8\u4efb\u52a1",
        "action.validate": "\u6821\u9a8c",
        "error.job_busy": "\u5df2\u6709\u4efb\u52a1\u6b63\u5728\u8fd0\u884c",
        "job.running": "\u6b63\u5728\u8fd0\u884c {name}",
        "label.language": "\u8bed\u8a00",
    },
}


def _format_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name.isidentifier() or format_spec or conversion:
            raise ValueError(f"Unsafe translation placeholder: {field_name!r}")
        fields.add(field_name)
    return fields


def validate_translations() -> None:
    """Reject incomplete dictionaries or incompatible format placeholders."""

    if set(TRANSLATIONS) != set(SUPPORTED_LANGUAGES):
        raise ValueError("Translations must define exactly en and zh-CN")
    expected_keys = set(TRANSLATIONS[DEFAULT_LANGUAGE])
    for language in SUPPORTED_LANGUAGES:
        translations = TRANSLATIONS[language]
        if set(translations) != expected_keys:
            raise ValueError(f"Translation keys do not match for {language}")
        for key, text in translations.items():
            if not isinstance(text, str) or not text:
                raise ValueError(f"Translation {language}.{key} must be non-empty text")
            if _format_fields(text) != _format_fields(
                TRANSLATIONS[DEFAULT_LANGUAGE][key]
            ):
                raise ValueError(
                    f"Translation placeholders do not match for {language}.{key}"
                )


@dataclass(frozen=True, slots=True)
class Translator:
    """Resolve and format user-facing text for one supported language."""

    language: str = DEFAULT_LANGUAGE

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported GUI language: {self.language!r}")

    def text(self, key: str, **values: object) -> str:
        """Return one formatted translation, raising ``KeyError`` if absent."""

        template = TRANSLATIONS[self.language][key]
        return template.format(**values)


validate_translations()
