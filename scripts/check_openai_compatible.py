from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.llm import OpenAICompatibleChatClient
from idea_graph.settings import OpenAICompatibleSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test an OpenAI-compatible chat endpoint.")
    parser.add_argument("--llm-config", type=Path, help="Path to a JSON config file.")
    parser.add_argument("--llm-base-url", help="Base URL for an OpenAI-compatible API endpoint.")
    parser.add_argument("--llm-model", help="Model name.")
    parser.add_argument("--llm-api-key", help="Direct API key.")
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="Environment variable for the API key when --llm-api-key is not provided.",
    )
    parser.add_argument(
        "--llm-provider",
        help="Provider family override such as openai or dashscope.",
    )
    parser.add_argument(
        "--llm-reasoning-mode",
        choices=["auto", "off", "on"],
        help="Reasoning mode hint for providers that expose a thinking toggle.",
    )
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--llm-max-tokens", type=int)
    parser.add_argument("--llm-timeout-seconds", type=int)
    parser.add_argument(
        "--prompt",
        default="Reply with a short JSON object like {\"status\":\"ok\",\"provider\":\"...\",\"model\":\"...\"}.",
        help="Prompt to send to the provider.",
    )
    return parser


def build_settings(args: argparse.Namespace) -> OpenAICompatibleSettings:
    payload: dict[str, object] = {}
    if args.llm_config:
        config_payload = json.loads(args.llm_config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise ValueError(f"LLM config file {args.llm_config} must contain a JSON object.")
        payload.update(config_payload)

    nested = payload.get("openai_compatible")
    if isinstance(nested, dict):
        merged = dict(nested)
    else:
        merged = dict(payload)

    if args.llm_base_url:
        merged["base_url"] = args.llm_base_url
    if args.llm_model:
        merged["model"] = args.llm_model
    if args.llm_api_key:
        merged["api_key"] = args.llm_api_key
    elif args.llm_api_key_env and os.getenv(args.llm_api_key_env):
        merged["api_key"] = os.getenv(args.llm_api_key_env)
    if args.llm_api_key_env:
        merged["api_key_env"] = args.llm_api_key_env
    if args.llm_provider:
        merged["provider"] = args.llm_provider
    if args.llm_reasoning_mode:
        merged["reasoning_mode"] = args.llm_reasoning_mode
    if args.llm_temperature is not None:
        merged["temperature"] = args.llm_temperature
    if args.llm_max_tokens is not None:
        merged["max_tokens"] = args.llm_max_tokens
    if args.llm_timeout_seconds is not None:
        merged["timeout_seconds"] = args.llm_timeout_seconds
    return OpenAICompatibleSettings.from_mapping(merged)


def main() -> None:
    args = build_parser().parse_args()
    settings = build_settings(args)
    client = OpenAICompatibleChatClient(settings)
    result = client.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": args.prompt},
        ],
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    print("== Endpoint ==")
    print(settings.base_url)
    print()

    print("== Model ==")
    print(settings.model)
    print()

    print("== Provider ==")
    print(settings.provider)
    print()

    print("== Reasoning Mode ==")
    print(settings.reasoning_mode)
    print()

    print("== Response ==")
    print(result.content)


if __name__ == "__main__":
    main()
