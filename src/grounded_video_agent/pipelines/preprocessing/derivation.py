from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from grounded_video_agent.capabilities._support import json_value


@dataclass(frozen=True, slots=True)
class DerivationSpec:
    producer_name: str
    producer_version: str
    parameters: Any

    def __post_init__(self) -> None:
        if not self.producer_name.strip() or not self.producer_version.strip():
            raise ValueError("derivation producer identity must not be empty")

    @property
    def key(self) -> str:
        payload = json.dumps(
            json_value(
                {
                    "producer_name": self.producer_name,
                    "producer_version": self.producer_version,
                    "parameters": self.parameters,
                }
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()
