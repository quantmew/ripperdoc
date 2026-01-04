"""Mock modules for programmatic execution sandbox.

This package provides safe mock replacements for dangerous system modules
used in the programmatic execution environment.
"""

from ripperdoc.core.programmatic_mocks.modules import (
    # Constants
    FORBIDDEN_MODULES,
    ALLOWED_MODULES,
    # Mock classes
    MockPath,
    MockEnviron,
    MockOsPathModule,
    MockOsModule,
    MockSysModule,
    MockSubprocessModule,
    MockShutilModule,
    MockPathlibModule,
    MockIoModule,
    MockGlobModule,
    MockTempfileModule,
    MockSocketModule,
    # Singleton instances
    MOCK_OS,
    MOCK_SYS,
    MOCK_SUBPROCESS,
    MOCK_SHUTIL,
    MOCK_PATHLIB,
    MOCK_IO,
    MOCK_GLOB,
    MOCK_TEMPFILE,
    MOCK_SOCKET,
    # Module registry
    MOCK_MODULES,
    # Dynamic mock factories
    create_dynamic_mock_os,
    create_dynamic_mock_glob,
    create_dynamic_mock_pathlib,
)

__all__ = [
    # Constants
    "FORBIDDEN_MODULES",
    "ALLOWED_MODULES",
    # Mock classes
    "MockPath",
    "MockEnviron",
    "MockOsPathModule",
    "MockOsModule",
    "MockSysModule",
    "MockSubprocessModule",
    "MockShutilModule",
    "MockPathlibModule",
    "MockIoModule",
    "MockGlobModule",
    "MockTempfileModule",
    "MockSocketModule",
    # Singleton instances
    "MOCK_OS",
    "MOCK_SYS",
    "MOCK_SUBPROCESS",
    "MOCK_SHUTIL",
    "MOCK_PATHLIB",
    "MOCK_IO",
    "MOCK_GLOB",
    "MOCK_TEMPFILE",
    "MOCK_SOCKET",
    # Module registry
    "MOCK_MODULES",
    # Dynamic mock factories
    "create_dynamic_mock_os",
    "create_dynamic_mock_glob",
    "create_dynamic_mock_pathlib",
]
