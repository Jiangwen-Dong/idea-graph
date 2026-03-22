from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.llm import OpenAICompatibleChatClient
from idea_graph.settings import OpenAICompatibleSettings


class OpenAICompatibleChatClientTests(unittest.TestCase):
    def test_dashscope_qwen3_disables_thinking_for_non_streaming_calls(self) -> None:
        settings = OpenAICompatibleSettings.from_mapping(
            {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "test-key",
                "model": "qwen3-8b",
                "provider": "dashscope",
                "reasoning_mode": "auto",
            }
        )
        client = OpenAICompatibleChatClient(settings)

        payload = client._build_payload(
            messages=[{"role": "user", "content": "hello"}],
            model=settings.model,
            temperature=0.2,
            max_tokens=128,
        )

        self.assertFalse(payload["enable_thinking"])

    def test_openai_payload_does_not_add_dashscope_thinking_toggle(self) -> None:
        settings = OpenAICompatibleSettings.from_mapping(
            {
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-key",
                "model": "gpt-4o-mini",
                "provider": "openai",
            }
        )
        client = OpenAICompatibleChatClient(settings)

        payload = client._build_payload(
            messages=[{"role": "user", "content": "hello"}],
            model=settings.model,
            temperature=0.2,
            max_tokens=128,
        )

        self.assertNotIn("enable_thinking", payload)

    def test_dashscope_reasoning_on_fails_for_non_streaming_client(self) -> None:
        settings = OpenAICompatibleSettings.from_mapping(
            {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "test-key",
                "model": "qwen3-8b",
                "provider": "dashscope",
                "reasoning_mode": "on",
            }
        )
        client = OpenAICompatibleChatClient(settings)

        with self.assertRaisesRegex(ValueError, "non-streaming client"):
            client._build_payload(
                messages=[{"role": "user", "content": "hello"}],
                model=settings.model,
                temperature=0.2,
                max_tokens=128,
            )

    def test_dashscope_always_thinking_models_fail_early(self) -> None:
        settings = OpenAICompatibleSettings.from_mapping(
            {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "test-key",
                "model": "deepseek-r1",
                "provider": "dashscope",
                "reasoning_mode": "auto",
            }
        )
        client = OpenAICompatibleChatClient(settings)

        with self.assertRaisesRegex(ValueError, "always-thinking"):
            client._build_payload(
                messages=[{"role": "user", "content": "hello"}],
                model=settings.model,
                temperature=0.2,
                max_tokens=128,
            )


if __name__ == "__main__":
    unittest.main()
