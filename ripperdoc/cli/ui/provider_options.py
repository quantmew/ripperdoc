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
        # === Major Cloud Providers ===
        ProviderOption(
            key="openai",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="gpt-4o-mini",
            model_suggestions=(
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "o1-preview",
                "o1-mini",
            ),
            default_api_base="https://api.openai.com/v1",
        ),
        ProviderOption(
            key="anthropic",
            protocol=ProviderType.ANTHROPIC,
            default_model="claude-3-5-sonnet-20241022",
            model_suggestions=(
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            ),
            default_api_base=None,
        ),
        ProviderOption(
            key="google",
            protocol=ProviderType.GEMINI,
            default_model="gemini-1.5-pro",
            model_suggestions=(
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ),
            default_api_base="https://generativelanguage.googleapis.com/v1beta",
        ),
        # === Aggregators & Open Router ===
        ProviderOption(
            key="openrouter",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="openai/gpt-4o-mini",
            model_suggestions=(
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-flash-1.5",
                "meta-llama/llama-3.1-70b-instruct",
            ),
            default_api_base="https://openrouter.ai/api/v1",
        ),
        ProviderOption(
            key="poe",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="gpt-4o",
            model_suggestions=(
                "gpt-4o",
                "claude-3.5-sonnet",
                "gemini-1.5-pro",
                "mistral-large",
            ),
            default_api_base="https://api.poe.com/v1",
        ),
        # === Chinese Providers ===
        ProviderOption(
            key="deepseek",
            protocol=ProviderType.ANTHROPIC,
            default_model="deepseek-chat",
            model_suggestions=(
                "deepseek-chat",
                "deepseek-reasoner",
            ),
            default_api_base="https://api.deepseek.com/v1",
        ),
        ProviderOption(
            key="zhipu",
            protocol=ProviderType.ANTHROPIC,
            default_model="glm-4-flash",
            model_suggestions=(
                "glm-4-plus",
                "glm-4-flash",
                "glm-4.7",
                "glm-4.6",
                "glm-4.5",
                "glm-4-air",
            ),
            default_api_base="https://open.bigmodel.cn/api/anthropic",
        ),
        ProviderOption(
            key="moonshot",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="moonshot-v1-auto",
            model_suggestions=(
                "moonshot-v1-auto",
                "kimi-k2-0711-preview",
                "kimi-k2-turbo-preview",
                "kimi-k2-thinking",
                "kimi-k2-0905-preview",
            ),
            default_api_base="https://api.moonshot.cn/v1",
        ),
        ProviderOption(
            key="volcengine",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="doubao-pro-32k",
            model_suggestions=(
                # Doubao Pro 系列
                "doubao-pro-32k",
                "doubao-pro-256k",
                "doubao-pro-32k-functioncall-241028",
                "doubao-pro-32k-character-241215",
                # Doubao 1.5 系列
                "Doubao-1.5-pro",
                "doubao-1.5-pro-32k",
                "doubao-1.5-pro-32k-character",
                "Doubao-1.5-pro-256k",
                "Doubao-1.5-vision-pro",
                "doubao-1.5-vision-pro",
                "Doubao-1.5-lite-32k",
                # Doubao Lite 系列
                "Doubao-lite-32k",
                "Doubao-lite-128k",
                "Doubao-lite-4k-character-240828",
                "Doubao-lite-32k-character-241015",
                # DeepSeek 系列
                "DeepSeek-V3",
                "DeepSeek-R1",
                "DeepSeek-R1-Distill-Qwen-32B",
                "DeepSeek-R1-Distill-Qwen-7B",
                # Vision 系列
                "Doubao-vision-lite-32k",
            ),
            default_api_base="https://ark.cn-beijing.volces.com/api/v3",
        ),
        ProviderOption(
            key="aliyun",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="qwen-plus",
            model_suggestions=(
                "qwen-plus",
                "qwen-turbo",
                "qwen-max",
                "qwen-coder-plus",
            ),
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        ProviderOption(
            key="minimax",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="abab6.5s",
            model_suggestions=(
                # abab 系列
                "abab6.5s",
                "abab6.5g",
                "abab6.5t",
                "abab6",
                "abab5.5s",
                "abab5",
                # 01 系列
                "minimax-01",
                # M2 系列
                "MiniMax-M2",
                "MiniMax-M2-Stable",
            ),
            default_api_base="https://api.minimax.chat/v1",
        ),
        ProviderOption(
            key="z.ai",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="glm-4-flash",
            model_suggestions=(
                "glm-4-flash",
                "glm-4-plus",
                "glm-4.6",
            ),
            default_api_base="https://api.z.ai/api/paas/v4",
        ),
        # === Western AI Companies ===
        ProviderOption(
            key="mistralai",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="mistral-large-latest",
            model_suggestions=(
                # Mistral Chat 系列
                "mistral-large-latest",
                "mistral-small-latest",
                "mistral-nemo",
                "mistral-mini",
                # 免费模型
                "mistral-7b",
                "mistral-8b",
                # Mistral Code 系列
                "codestral-latest",
                # 多模态
                "pixtral-large-latest",
            ),
            default_api_base="https://api.mistral.ai/v1",
        ),
        ProviderOption(
            key="groq",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="llama-3.3-70b-versatile",
            model_suggestions=(
                # Llama 系列
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "llama3-8b-8192",
                # Gemma 系列
                "gemma2-9b-it",
                "gemma-7b-it",
                # Mistral 系列
                "mistral-saba-24b",
                "mixtral-8x7b-32768",
            ),
            default_api_base="https://api.groq.com/openai/v1",
        ),
        ProviderOption(
            key="grok",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="grok-3",
            model_suggestions=(
                "grok-4",
                "grok-3",
                "grok-3-fast",
                "grok-3-mini",
                "grok-3-mini-fast",
            ),
            default_api_base="https://api.x.ai/v1",
        ),
        ProviderOption(
            key="cohere",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="command-r-plus-08-2024",
            model_suggestions=(
                "command-r-plus-08-2024",
                "command-r-08-2024",
                "command-r7b-12-2024",
            ),
            default_api_base="https://api.cohere.ai/v1",
        ),
        ProviderOption(
            key="together",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            model_suggestions=(
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
            ),
            default_api_base="https://api.together.xyz/v1",
        ),
        ProviderOption(
            key="perplexity",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="llama-3.1-sonar-small-128k-online",
            model_suggestions=(
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online",
            ),
            default_api_base="https://api.perplexity.ai",
        ),
        ProviderOption(
            key="siliconflow",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="Qwen/Qwen2.5-72B-Instruct",
            model_suggestions=(
                "Qwen/Qwen2.5-72B-Instruct",
                "deepseek-ai/DeepSeek-V3",
                "01-ai/Yi-1.5-34B-Chat",
            ),
            default_api_base="https://api.siliconflow.cn/v1",
        ),
        # === Generic / Custom ===
        ProviderOption(
            key="openai_compatible",
            protocol=ProviderType.OPENAI_COMPATIBLE,
            default_model="gpt-4o-mini",
            model_suggestions=(
                "gpt-4o-mini",
                "gpt-4o",
                "llama-3.1-70b",
            ),
            default_api_base=None,
        ),
        ProviderOption(
            key="anthropic_compatible",
            protocol=ProviderType.ANTHROPIC,
            default_model="claude-3-5-sonnet-20241022",
            model_suggestions=(
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ),
            default_api_base=None,
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
