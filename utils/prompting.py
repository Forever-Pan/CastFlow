"""Validated external prompt packs and explicitly generic reference templates."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from string import Formatter
from typing import Any


PROMPT_ROLES = (
    "forecast_system",
    "forecast_user",
    "planning_system",
    "planning_user",
    "reflection_system",
    "reflection_user",
    "exploration_system",
    "exploration_user",
)
PROMPT_FORMAT_VERSION = "castflow-prompt-pack-v1"
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_DIRECTORY = _SOURCE_ROOT / "forecasting" / "reference_prompts"
_MAX_TEMPLATE_BYTES = 1024 * 1024


class PromptConfigurationError(ValueError):
    """An unusable prompt pack; errors never include private template contents."""


@dataclass(frozen=True, slots=True)
class PromptPack:
    mode: str
    fingerprint: str
    templates: dict[str, str] = field(repr=False)
    role_hashes: dict[str, str]

    def identity(self) -> dict[str, Any]:
        return {
            "format_version": PROMPT_FORMAT_VERSION,
            "mode": self.mode,
            "fingerprint": self.fingerprint,
            "roles": dict(self.role_hashes),
        }


@dataclass(slots=True)
class _PromptConfig:
    prompt_dir: str | None = None
    prompt_mode: str | None = None
    prompt_fingerprint: str | None = None


def ensure_prompt_config(config: Any = None) -> Any:
    """Create a prompt-only configuration when no workflow config is supplied."""
    if config is None:
        from utils.config import load_castflow_env

        values = load_castflow_env()
        config = _PromptConfig(prompt_dir=values.get("CASTFLOW_PROMPT_DIR") or None)
    validate_prompt_config(config)
    return config


def _template_fields(template: str, role: str) -> None:
    try:
        fields = []
        for _, name, spec, conversion in Formatter().parse(template):
            if name is not None:
                if name != "context" or spec or conversion:
                    raise PromptConfigurationError(
                        f"{role}: unsupported template placeholder"
                    )
                fields.append(name)
    except ValueError:
        raise PromptConfigurationError(f"{role}: invalid template formatting") from None
    expected = ["context"] if role.endswith("_user") else []
    if fields != expected:
        requirement = (
            "exactly one context placeholder" if expected else "no placeholders"
        )
        raise PromptConfigurationError(f"{role}: template requires {requirement}")


def load_prompt_pack(prompt_dir: str | Path | None = None) -> PromptPack:
    """Read a complete pack without executing templates or searching for secrets.

    A custom pack must reside outside the installed source tree. With no custom
    directory, only packaged reference templates are used; no private location
    is guessed or loaded from a nearby checkout.
    """
    mode = "reference" if prompt_dir is None else "custom"
    if prompt_dir is not None and (
        isinstance(prompt_dir, bool)
        or not isinstance(prompt_dir, (str, Path))
        or not str(prompt_dir).strip()
    ):
        raise PromptConfigurationError("prompt_dir must name an external directory")
    try:
        directory = (
            _REFERENCE_DIRECTORY
            if prompt_dir is None
            else Path(prompt_dir).expanduser()
        ).resolve(strict=True)
    except (OSError, RuntimeError):
        raise PromptConfigurationError(
            "prompt directory is missing or inaccessible"
        ) from None
    if not directory.is_dir():
        raise PromptConfigurationError("prompt_dir must be a directory")
    if mode == "custom" and directory.is_relative_to(_SOURCE_ROOT):
        raise PromptConfigurationError(
            "custom prompts must be stored outside the source checkout"
        )
    expected = {role + ".txt" for role in PROMPT_ROLES}
    try:
        found = {path.name for path in directory.iterdir() if path.suffix == ".txt"}
    except OSError:
        raise PromptConfigurationError("prompt directory is inaccessible") from None
    if found != expected:
        raise PromptConfigurationError(
            "prompt pack must contain exactly the eight supported role files"
        )
    templates, hashes = {}, {}
    for role in PROMPT_ROLES:
        path = directory / (role + ".txt")
        try:
            resolved = path.resolve(strict=True)
            if mode == "custom" and resolved.is_relative_to(_SOURCE_ROOT):
                raise PromptConfigurationError(
                    f"{role}: custom template resolves inside the source checkout"
                )
            if not path.is_file() or path.stat().st_size > _MAX_TEMPLATE_BYTES:
                raise PromptConfigurationError(
                    f"{role}: template is missing or too large"
                )
            raw = path.read_bytes()
            if len(raw) > _MAX_TEMPLATE_BYTES:
                raise PromptConfigurationError(f"{role}: template is too large")
            template = raw.decode("utf-8")
        except (OSError, UnicodeError):
            raise PromptConfigurationError(
                f"{role}: template must be a readable UTF-8 file"
            ) from None
        if not template.strip() or "\x00" in template:
            raise PromptConfigurationError(
                f"{role}: template must contain nonempty text"
            )
        _template_fields(template, role)
        templates[role] = template
        hashes[role] = hashlib.sha256(raw).hexdigest()
    payload = {"format_version": PROMPT_FORMAT_VERSION, "mode": mode, "roles": hashes}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PromptPack(mode, fingerprint, templates, hashes)


def validate_prompt_config(config: Any) -> PromptPack:
    """Seal a configuration to one content identity before making model calls."""
    pack = load_prompt_pack(config.prompt_dir)
    previous = getattr(config, "prompt_fingerprint", None)
    previous_mode = getattr(config, "prompt_mode", None)
    if previous is not None and previous != pack.fingerprint:
        raise PromptConfigurationError(
            "prompt pack changed after configuration validation; create a new configuration"
        )
    if previous_mode is not None and previous_mode != pack.mode:
        raise PromptConfigurationError(
            "prompt mode differs from the validated configuration"
        )
    config.prompt_fingerprint = pack.fingerprint
    config.prompt_mode = pack.mode
    if config.prompt_dir is not None:
        config.prompt_dir = str(
            Path(config.prompt_dir).expanduser().resolve(strict=True)
        )
    return pack


def prompt_identity(config: Any = None) -> dict[str, Any]:
    """Return hashes and the reference/custom mode, never template text."""
    pack = validate_prompt_config(ensure_prompt_config(config))
    return pack.identity()


def render_prompt(config: Any, role: str, *, context: str = "") -> str:
    """Insert one data context without recursive formatting or object access."""
    if role not in PROMPT_ROLES:
        raise PromptConfigurationError("unsupported prompt role")
    if not isinstance(context, str):
        raise PromptConfigurationError("prompt context must be text")
    if role.endswith("_system") and context:
        raise PromptConfigurationError("system prompt roles do not accept context")
    pack = validate_prompt_config(ensure_prompt_config(config))
    try:
        return pack.templates[role].format_map({"context": context})
    except (ValueError, KeyError, IndexError, AttributeError):
        raise PromptConfigurationError(f"{role}: template rendering failed") from None
