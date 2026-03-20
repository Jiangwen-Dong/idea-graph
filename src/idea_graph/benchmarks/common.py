from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_CHUNK_SIZE = 1024 * 1024
USER_AGENT = "idea-graph/0.1 (+local benchmark downloader)"


def user_agent_request(url: str) -> Request:
    return Request(url, headers={"User-Agent": USER_AGENT})


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent)) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        with urlopen(user_agent_request(url), timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=DEFAULT_CHUNK_SIZE)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
