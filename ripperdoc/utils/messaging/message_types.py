"""
Type aliases for messaging module.

This module provides clear type distinctions between:
1. All conversation messages (including AttachmentMessage)
2. Model-visible messages (excluding AttachmentMessage, which should be expanded)
"""

from typing import Union, List

from ripperdoc.utils.messaging.messages import (
    UserMessage,
    AssistantMessage,
    ProgressMessage,
    AttachmentMessage,
)

# All message types that can appear in conversation history
# Includes AttachmentMessage which needs to be expanded before sending to model
ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage, AttachmentMessage]

# Model-visible message types (AttachmentMessage must be expanded before use)
# This type represents messages that can be directly sent to AI providers
ModelMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

# Type aliases for lists
ConversationMessageList = List[ConversationMessage]
ModelMessageList = List[ModelMessage]
