"""Permission and tool-call protocol DTOs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class ToolCallRequest(BaseModel):
    """Request payload for MCP-style tool invocations."""

    name: str
    arguments: Optional[dict[str, Any]] = None
    meta: Optional[dict[str, Any]] = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        protected_namespaces=(),
    )


class PermissionResponseAllow(BaseModel):
    """A permission allow response."""

    behavior: str = Field(default="allow")
    updatedInput: Optional[dict[str, Any]] = None
    toolUseID: Optional[str] = None
    decisionReason: Optional[dict[str, Any]] = None
    updatedPermissions: Optional[List[Dict[str, Any]]] = None


class PermissionResponseDeny(BaseModel):
    """A permission deny response."""

    behavior: str = Field(default="deny")
    message: str = ""
    toolUseID: Optional[str] = None
    decisionReason: Optional[dict[str, Any]] = None


class PermissionRequestPayload(BaseModel):
    """Payload for SDK can_use_tool permission requests."""

    subtype: str = Field(default="can_use_tool")
    tool_name: str
    input: Optional[dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    agent_id: Optional[str] = None
    permission_suggestions: Optional[List[Dict[str, Any]]] = None
    blocked_path: Optional[str] = None
    decision_reason: Optional[dict[str, Any]] = None
    force_prompt: bool = False


class PermissionRuleValue(BaseModel):
    """A single permission rule with tool name and content."""

    tool_name: str = Field(alias="toolName")
    rule_content: Optional[str] = Field(default=None, alias="ruleContent")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateDestination(str):
    """Where to apply a permission update."""

    USER_SETTINGS = "userSettings"
    PROJECT_SETTINGS = "projectSettings"
    LOCAL_SETTINGS = "localSettings"
    SESSION = "session"
    CLI_ARG = "cliArg"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.str_schema()


class PermissionDecisionClassification(str):
    """Classification of a permission decision."""

    USER_TEMPORARY = "user_temporary"
    USER_PERMANENT = "user_permanent"
    USER_REJECT = "user_reject"


class PermissionUpdateAddRules(BaseModel):
    """Permission update: add rules."""

    type: Literal["addRules"] = "addRules"
    rules: list[PermissionRuleValue]
    behavior: Literal["allow", "deny", "ask"] = "allow"
    destination: Optional[PermissionUpdateDestination] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateReplaceRules(BaseModel):
    """Permission update: replace all rules."""

    type: Literal["replaceRules"] = "replaceRules"
    rules: list[PermissionRuleValue]
    behavior: Literal["allow", "deny", "ask"] = "allow"
    destination: Optional[PermissionUpdateDestination] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateRemoveRules(BaseModel):
    """Permission update: remove rules."""

    type: Literal["removeRules"] = "removeRules"
    rules: list[PermissionRuleValue]
    behavior: Literal["allow", "deny", "ask"] = "allow"
    destination: Optional[PermissionUpdateDestination] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateSetMode(BaseModel):
    """Permission update: set permission mode."""

    type: Literal["setMode"] = "setMode"
    mode: str
    destination: Optional[PermissionUpdateDestination] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateAddDirectories(BaseModel):
    """Permission update: add working directories."""

    type: Literal["addDirectories"] = "addDirectories"
    directories: list[str]
    destination: Optional[PermissionUpdateDestination] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateRemoveDirectories(BaseModel):
    """Permission update: remove working directories."""

    type: Literal["removeDirectories"] = "removeDirectories"
    directories: list[str]
    destination: Optional[PermissionUpdateDestination] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


PermissionUpdate = Union[
    PermissionUpdateAddRules,
    PermissionUpdateReplaceRules,
    PermissionUpdateRemoveRules,
    PermissionUpdateSetMode,
    PermissionUpdateAddDirectories,
    PermissionUpdateRemoveDirectories,
]


__all__ = [
    "ToolCallRequest",
    "PermissionResponseAllow",
    "PermissionResponseDeny",
    "PermissionRequestPayload",
    "PermissionRuleValue",
    "PermissionUpdateDestination",
    "PermissionDecisionClassification",
    "PermissionUpdateAddRules",
    "PermissionUpdateReplaceRules",
    "PermissionUpdateRemoveRules",
    "PermissionUpdateSetMode",
    "PermissionUpdateAddDirectories",
    "PermissionUpdateRemoveDirectories",
    "PermissionUpdate",
]
