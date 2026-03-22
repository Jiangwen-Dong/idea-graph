from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .settings import OpenAICompatibleSettings


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    model: str
    raw_response: dict[str, Any]


class OpenAICompatibleChatClient:
    def __init__(self, settings: OpenAICompatibleSettings) -> None:
        self.settings = settings

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletionResult:
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_text = self._post_json(payload)
        raw_response = json.loads(raw_text)
        content = self._extract_content(raw_response)
        return ChatCompletionResult(
            content=content,
            model=str(payload["model"]),
            raw_response=raw_response,
        )

    def _build_payload(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model or self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if self.settings.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.settings.extra_body:
            payload.update(self.settings.extra_body)
        self._apply_provider_adaptations(payload)
        return payload

    def _post_json(self, payload: dict[str, object]) -> str:
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.settings.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
            **self.settings.extra_headers,
        }

        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                self._format_http_error(payload=payload, status_code=exc.code, error_text=error_text)
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc

    def _apply_provider_adaptations(self, payload: dict[str, object]) -> None:
        if self._provider_family() != "dashscope":
            return

        model_name = self._normalized_model_name(str(payload.get("model", "")))
        reasoning_mode = self.settings.reasoning_mode

        if self._is_dashscope_always_thinking_model(model_name):
            raise ValueError(
                "DashScope model "
                f"'{payload.get('model')}' appears to be an always-thinking model. "
                "The current client is non-streaming, so it cannot safely force thinking off. "
                "Use a mixed-thinking model such as qwen3-8b or qwen-plus, or extend the client with streaming support."
            )

        if not self._supports_dashscope_reasoning_toggle(model_name):
            return

        if reasoning_mode == "on":
            raise ValueError(
                "reasoning_mode='on' is not supported by the current non-streaming client for DashScope reasoning models. "
                "Set reasoning_mode to 'auto' or 'off', or add streaming support."
            )

        if reasoning_mode in {"auto", "off"}:
            payload["enable_thinking"] = False

    def _format_http_error(
        self,
        *,
        payload: dict[str, object],
        status_code: int,
        error_text: str,
    ) -> str:
        message = f"OpenAI-compatible request failed with status {status_code}: {error_text}"
        if self._provider_family() != "dashscope":
            return message

        lowered = error_text.lower()
        if "parameter.enable_thinking must be set to false for non-streaming calls" in lowered:
            return (
                f"{message}\n"
                "DashScope hint: this client uses non-streaming chat completions. "
                "For Qwen reasoning-capable models, set provider='dashscope' and reasoning_mode='auto' or 'off' "
                "so the client can send enable_thinking=false automatically."
            )
        if "enable_thinking" in lowered and self._is_dashscope_always_thinking_model(
            self._normalized_model_name(str(payload.get("model", "")))
        ):
            return (
                f"{message}\n"
                "DashScope hint: this model appears to be always-thinking on DashScope. "
                "The current client only supports non-streaming requests, so choose a model with switchable thinking "
                "or add streaming support."
            )
        return message

    def _provider_family(self) -> str:
        provider = self.settings.provider
        if provider != "auto":
            return provider
        base_url = self.settings.base_url.lower()
        if "dashscope.aliyuncs.com" in base_url:
            return "dashscope"
        if "api.openai.com" in base_url:
            return "openai"
        return "openai-compatible"

    @staticmethod
    def _normalized_model_name(model: str) -> str:
        return model.strip().lower()

    @classmethod
    def _supports_dashscope_reasoning_toggle(cls, model_name: str) -> bool:
        if cls._is_dashscope_always_thinking_model(model_name):
            return False
        return any(
            marker in model_name
            for marker in (
                "qwen3",
                "qwen-plus",
                "qwen-max",
                "qwen-turbo",
            )
        )

    @staticmethod
    def _is_dashscope_always_thinking_model(model_name: str) -> bool:
        return any(
            marker in model_name
            for marker in (
                "qwq",
                "deepseek-r1",
                "-thinking",
                "thinking-",
            )
        )

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-compatible response did not contain any choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if isinstance(text, str):
                        text_parts.append(text)
            joined = "\n".join(part.strip() for part in text_parts if part.strip())
            if joined:
                return joined

        raise ValueError("OpenAI-compatible response did not contain a supported message content field.")
