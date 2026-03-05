"""Remote-control bridge components.

This package implements a split architecture:
- API client
- bridge loop runner
- token/session refresh manager
- sessions websocket manager
- REPL bridge manager
"""

from .api import RemoteControlApiClient
from .errors import BridgeFatalError
from .loop import ActiveSession, RemoteControlBridgeRunner
from .models import RegisteredEnvironment, RemoteControlConfig, WorkSecret
from .process import BridgeActivity, ChildBridgeSession, RemoteControlProcessSpawner
from .repl_bridge import RemoteSessionBridgeManager, RemoteSessionCallbacks, RemoteSessionConfig
from .session_ws import SessionsWebSocketManager
from .token_manager import TokenSessionManager
from .utils import build_session_ingress_ws_url, decode_work_secret

__all__ = [
    "ActiveSession",
    "BridgeFatalError",
    "BridgeActivity",
    "ChildBridgeSession",
    "RegisteredEnvironment",
    "RemoteControlApiClient",
    "RemoteControlBridgeRunner",
    "RemoteControlConfig",
    "RemoteControlProcessSpawner",
    "RemoteSessionBridgeManager",
    "RemoteSessionCallbacks",
    "RemoteSessionConfig",
    "SessionsWebSocketManager",
    "TokenSessionManager",
    "WorkSecret",
    "build_session_ingress_ws_url",
    "decode_work_secret",
]
