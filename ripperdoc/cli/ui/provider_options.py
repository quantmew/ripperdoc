"""
Provider metadata used by interactive UI flows.

Each provider option represents exactly one protocol endpoint. Vendors that
offer multiple protocols (for example, DeepSeek exposing both OpenAI-style and
Anthropic-style APIs on different URLs) should be modeled as distinct provider
options so that protocol-specific defaults stay unambiguous.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ripperdoc.core.config import ProviderType


@dataclass(frozen=True)
class ProviderOption:
    """A single provider choice with protocol and UX defaults."""

    key: str
    protocol: ProviderType
    default_model: str
    model_suggestions: Tuple[str, ...] = ()
    default_api_base: Optional[str] = None

    @property
    def label(self) -> str:
        """Display label shown to the user."""
        return self.key

    def with_api_base(self, api_base: Optional[str]) -> "ProviderOption":
        """Return a copy that overrides the default API base."""
        return ProviderOption(
            key=self.key,
            protocol=self.protocol,
            default_model=self.default_model,
            model_suggestions=self.model_suggestions,
            default_api_base=api_base if api_base else self.default_api_base,
        )


class ProviderRegistry:
    """Registry for known providers to keep wizard logic declarative."""

    def __init__(self, providers: Sequence[ProviderOption], default_key: str) -> None:
        if not providers:
            raise ValueError("Provider registry cannot be empty")
        self._providers: List[ProviderOption] = list(providers)
        self._index: Dict[str, ProviderOption] = {p.key: p for p in providers}
        if default_key not in self._index:
            raise ValueError(f"Default provider '{default_key}' not found in registry")
        self._default_key = default_key

    @property
    def providers(self) -> List[ProviderOption]:
        """Return providers in display order."""
        return list(self._providers)

    @property
    def default_choice(self) -> ProviderOption:
        """Return the default provider selection."""
        return self._index[self._default_key]

    def get(self, key: str) -> Optional[ProviderOption]:
        """Look up a provider option by key."""
        return self._index.get(key)

    def keys(self) -> List[str]:
        """Return provider keys in display order."""
        return [p.key for p in self._providers]


def default_model_for_protocol(protocol: ProviderType) -> str:
    """Reasonable default model per protocol family."""
    if protocol == ProviderType.ANTHROPIC:
        return "claude-3-5-sonnet-20241022"
    if protocol == ProviderType.GEMINI:
        return "gemini-1.5-pro"
    return "gpt-4o-mini"


KNOWN_PROVIDERS = ProviderRegistry(
    providers=[
        ProviderOption(
            key="deepseek",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="deepseek-chat",
            model_suggestions=("deepseek-chat", "deepseek-reasoner"),
            default_api_base="https://api.deepseek.com/v1",
        ),
        ProviderOption(
            key="openai",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="gpt-4o-mini",
            model_suggestions=(
                "gpt-5.1",
                "gpt-5.1-chat",
                "gpt-5.1-codex",
                "gpt-4o",
                "gpt-4-turbo",
                "o1-preview",
                "o1-mini",
            ),
            default_api_base="https://api.openai.com/v1",
        ),
        ProviderOption(
            key="openrouter",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="openai/gpt-4o-mini",
            model_suggestions=(
                "openai/gpt-4o-mini",
                "meta-llama/llama-3.1-8b-instruct",
                "google/gemini-flash-1.5",
            ),
            default_api_base="https://openrouter.ai/api/v1",
        ),
        ProviderOption(
            key="anthropic",
            protocol=ProviderType.ANTHROPIC,
            default_model="claude-3-5-sonnet-20241022",
            model_suggestions=(
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ),
            default_api_base=None,
        ),
        ProviderOption(
            key="openai_compatible",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="gpt-4o-mini",
            model_suggestions=(
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-3.5-turbo",
            ),
            default_api_base=None,
        ),
        ProviderOption(
            key="mistralai",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="mistral-small-creative",
            model_suggestions=(
                "mistral-small-creative",
                "mistral-large-latest",
                "mistral-small-latest",
                "devstral-2512",
                "ministral-14b-2512",
                "ministral-8b-2512",
                "codestral-latest",
                "pixtral-large-latest",
            ),
            default_api_base="https://api.mistral.ai/v1",
        ),
        ProviderOption(
            key="google",
            protocol=ProviderType.GEMINI,
            default_model="gemini-1.5-pro",
            model_suggestions=(
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash",
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
            ),
            default_api_base="https://generativelanguage.googleapis.com/v1beta",
        ),
        ProviderOption(
            key="moonshot",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="kimi-k2-turbo-preview",
            model_suggestions=(
                "kimi-k2-0905-preview",
                "kimi-k2-0711-preview",
                "kimi-k2-turbo-preview",
                "kimi-k2-thinking",
                "kimi-k2-thinking-turbo",
            ),
            default_api_base="https://api.moonshot.cn/v1",
        ),
        ProviderOption(
            key="qwen",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="qwen-turbo",
            model_suggestions=(
                "qwen-turbo",
                "qwen-plus",
                "qwen-max",
                "qwen2.5-32b",
                "qwen2.5-coder-32b",
            ),
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        ProviderOption(
            key="zhipu",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="glm-4-flash",
            model_suggestions=(
                "glm-4-plus",
                "glm-4-air-250414",
                "glm-4-airx",
                "glm-4-long",
                "glm-4-flashx",
                "glm-4-flash-250414",
                "glm-4.6",
                "glm-4.5",
                "glm-4.5-air",
                "glm-4.5-airx",
                "glm-4.5-x",
                "glm-4.5-flash",
            ),
            default_api_base="https://open.bigmodel.cn/api/paas/v4",
        ),
        ProviderOption(
            key="minimax",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="MiniMax-M2",
            model_suggestions=("MiniMax-M2",),
            default_api_base="https://api.minimax.chat/v1",
        ),
        ProviderOption(
            key="siliconflow",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="deepseek-ai/DeepSeek-V3.2",
            model_suggestions=(
                "deepseek-ai/DeepSeek-V3.2",
                "Qwen/Qwen2.5-32B-Instruct",
                "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                "zai-org/GLM-4.6",
                "moonshotai/Kimi-K2-Thinking",
                "MiniMaxAI/MiniMax-M2",
            ),
            default_api_base="https://api.siliconflow.cn/v1",
        ),
    ],
    default_key="deepseek",
)


__all__ = [
    "KNOWN_PROVIDERS",
    "ProviderOption",
    "ProviderRegistry",
    "default_model_for_protocol",
]
