from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


VALID_REASONING_MODES = {"auto", "off", "on"}


def _normalize_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("openai_compatible")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(payload)


def _ensure_string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        normalized[str(key)] = str(item)
    return normalized


def _looks_like_api_key(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("sk-") or stripped.startswith("dashscope-")


def _ensure_object_dict(value: Any) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    base_url: str
    api_key: str
    model: str
    provider: str = "auto"
    reasoning_mode: str = "auto"
    temperature: float = 0.2
    max_tokens: int = 1400
    timeout_seconds: int = 90
    max_retries: int = 2
    api_key_env: str = "OPENAI_API_KEY"
    role_models: dict[str, str] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, object] = field(default_factory=dict)
    json_mode: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "OpenAICompatibleSettings":
        data = _normalize_mapping(payload)
        base_url = str(data.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        raw_api_key_env = str(data.get("api_key_env", "OPENAI_API_KEY")).strip()
        direct_api_key = str(data.get("api_key", "")).strip()
        inline_api_key = raw_api_key_env if _looks_like_api_key(raw_api_key_env) else ""
        api_key_env = raw_api_key_env if raw_api_key_env and not inline_api_key else "INLINE_API_KEY"
        api_key = str(
            direct_api_key
            or inline_api_key
            or os.getenv(raw_api_key_env)
            or os.getenv("IDEA_GRAPH_API_KEY")
            or ""
        ).strip()
        model = str(data.get("model") or os.getenv("IDEA_GRAPH_MODEL") or "").strip()
        if not api_key:
            raise ValueError(
                f"Missing API key for OpenAI-compatible backend. Set '{raw_api_key_env}' or provide 'api_key' in the config."
            )
        if not model:
            raise ValueError("Missing model for OpenAI-compatible backend.")
        provider = str(data.get("provider", "auto")).strip().lower() or "auto"
        reasoning_mode = str(data.get("reasoning_mode", "auto")).strip().lower() or "auto"
        if reasoning_mode not in VALID_REASONING_MODES:
            valid = ", ".join(sorted(VALID_REASONING_MODES))
            raise ValueError(f"Invalid reasoning_mode '{reasoning_mode}'. Expected one of: {valid}.")

        role_models = {
            key: value
            for key, value in _ensure_string_dict(data.get("role_models", {})).items()
            if value and not value.startswith("your-")
        }

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            provider=provider,
            reasoning_mode=reasoning_mode,
            temperature=float(data.get("temperature", 0.2)),
            max_tokens=int(data.get("max_tokens", 1400)),
            timeout_seconds=int(data.get("timeout_seconds", 90)),
            max_retries=int(data.get("max_retries", 2)),
            api_key_env=api_key_env,
            role_models=role_models,
            extra_headers=_ensure_string_dict(data.get("extra_headers", {})),
            extra_body=_ensure_object_dict(data.get("extra_body", {})),
            json_mode=bool(data.get("json_mode", False)),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "OpenAICompatibleSettings":
        file_path = Path(path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"LLM config file {file_path} must contain a JSON object.")
        return cls.from_mapping(payload)

    def model_for_role(self, role: str) -> str:
        return self.role_models.get(role, self.model)

    def sanitized_dict(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "provider": self.provider,
            "reasoning_mode": self.reasoning_mode,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "api_key_env": self.api_key_env,
            "role_models": self.role_models,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "json_mode": self.json_mode,
        }


@dataclass(frozen=True)
class AgentRuntimeConfig:
    backend: str = "deterministic"
    fallback_to_deterministic: bool = True
    openai_compatible: OpenAICompatibleSettings | None = None

    def uses_llm(self) -> bool:
        return self.backend == "openai-compatible" and self.openai_compatible is not None
