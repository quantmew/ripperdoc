"""Provider catalog models shared across UI and config flows."""

from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict

from ripperdoc.core.config import ModelProfile, ProtocolType


class Provider(BaseModel):
    """Provider metadata independent of UI concerns."""

    model_config = ConfigDict(frozen=True)

    protocol: ProtocolType
    website: str
    description: str
    base_url: Optional[str] = None
    model_list: Tuple[ModelProfile, ...] = ()
