#!/usr/bin/env python3
"""Send desktop notifications.

This hook sends desktop notifications when Ripperdoc notifications occur.
Works on Linux (notify-send), macOS (osascript), and Windows (toast).

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/log_notification.py"
          }
        ]
      }
    ]
  }
}
"""

import json
import platform
import subprocess
import sys


def send_notification_linux(title: str, message: str) -> bool:
    """Send notification on Linux using notify-send."""
    try:
        subprocess.run(
            ["notify-send", title, message],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def send_notification_macos(title: str, message: str) -> bool:
    """Send notification on macOS using osascript."""
    try:
        script = f'display notification "{message}" with title "{title}" sound name "default"'
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def send_notification_windows(title: str, message: str) -> bool:
    """Send notification on Windows using PowerShell toast."""
    try:
        # Use PowerShell to send a toast notification
        ps_script = f'''
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

        $template = @"
        <toast>
            <visual>
                <binding template="ToastText02">
                    <text id="1">{title}</text>
                    <text id="2">{message}</text>
                </binding>
            </visual>
        </toast>
"@

        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml($template)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Ripperdoc").Show($toast)
        '''
        subprocess.run(
            ["powershell", "-Command", ps_script],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def send_notification(title: str, message: str) -> bool:
    """Send a desktop notification based on the current platform."""
    system = platform.system()

    if system == "Linux":
        return send_notification_linux(title, message)
    elif system == "Darwin":
        return send_notification_macos(title, message)
    elif system == "Windows":
        return send_notification_windows(title, message)
    else:
        return False


def main() -> None:
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    title = input_data.get("title", "Ripperdoc")
    message = input_data.get("message", "")
    notification_type = input_data.get("notification_type", "info")

    if not message:
        sys.exit(0)

    # Add emoji based on type
    type_emojis = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅",
    }
    emoji = type_emojis.get(notification_type, "")
    if emoji:
        title = f"{emoji} {title}"

    # Send the notification
    success = send_notification(title, message)

    if not success:
        print("Could not send desktop notification", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
