@echo off
setlocal EnableExtensions EnableDelayedExpansion

where powershell >nul 2>&1
if errorlevel 1 (
    echo Error: PowerShell is required to run this installer.
    exit /b 1
)

set "PS_ARGS="
:parse
if "%~1"=="" goto run

if /i "%~1"=="--uninstall" (
    set "PS_ARGS=%PS_ARGS% -Uninstall"
    shift
    goto parse
)

if /i "%~1"=="--bin-dir" (
    if "%~2"=="" (
        echo Usage: install.cmd [--uninstall] [--bin-dir DIR] [latest|stable|VERSION]
        exit /b 1
    )
    set "PS_ARGS=%PS_ARGS% -BinDir \"%~2\""
    shift
    shift
    goto parse
)

set "PS_ARGS=%PS_ARGS% %~1"
shift
goto parse

:run
if defined PS_ARGS (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %PS_ARGS%
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
)

set "RC=%ERRORLEVEL%"
if defined PS_ARGS (set "PS_ARGS=")
exit /b %RC%
