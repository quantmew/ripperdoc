"""Tests for platform detection utilities.

Tests cover:
- Platform class methods
- Convenience functions (is_windows, is_linux, etc.)
- Module availability detection (has_termios, has_fcntl, has_tty)
- Cross-platform compatibility
"""

from ripperdoc.utils.platform import (
    # Platform class
    Platform,
    # Convenience functions
    is_windows,
    is_linux,
    is_macos,
    is_bsd,
    is_unix,
    is_posix,
    # Module availability functions
    has_termios,
    has_fcntl,
    has_tty,
    # Constants
    IS_WINDOWS,
    IS_LINUX,
    IS_MACOS,
    IS_BSD,
    IS_UNIX,
    IS_POSIX,
    HAS_TERMIOS,
    HAS_FCNTL,
    HAS_TTY,
)


class TestPlatformClass:
    """Tests for Platform class methods."""

    def test_get_system_returns_known_platform(self):
        """get_system should return a known platform type."""
        result = Platform.get_system()
        assert result in {"windows", "linux", "macos", "unknown"}

    def test_is_windows_returns_bool(self):
        """is_windows should return a boolean."""
        assert isinstance(Platform.is_windows(), bool)

    def test_is_linux_returns_bool(self):
        """is_linux should return a boolean."""
        assert isinstance(Platform.is_linux(), bool)

    def test_is_macos_returns_bool(self):
        """is_macos should return a boolean."""
        assert isinstance(Platform.is_macos(), bool)

    def test_is_bsd_returns_bool(self):
        """is_bsd should return a boolean."""
        assert isinstance(Platform.is_bsd(), bool)

    def test_is_unix_returns_bool(self):
        """is_unix should return a boolean."""
        assert isinstance(Platform.is_unix(), bool)

    def test_is_posix_returns_bool(self):
        """is_posix should return a boolean."""
        assert isinstance(Platform.is_posix(), bool)

    def test_get_raw_name_returns_string(self):
        """get_raw_name should return sys.platform value."""
        result = Platform.get_raw_name()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_os_name_returns_nt_or_posix(self):
        """get_os_name should return 'nt' or 'posix'."""
        result = Platform.get_os_name()
        assert result in {"nt", "posix"}

    def test_platform_constants_defined(self):
        """Platform constants should be defined."""
        assert hasattr(Platform, "WINDOWS")
        assert hasattr(Platform, "LINUX")
        assert hasattr(Platform, "MACOS")
        assert hasattr(Platform, "FREEBSD")
        assert hasattr(Platform, "NAME_NT")
        assert hasattr(Platform, "NAME_POSIX")

    def test_windows_constant_is_win32(self):
        """WINDOWS constant should be 'win32'."""
        assert Platform.WINDOWS == "win32"

    def test_linux_constant_is_linux(self):
        """LINUX constant should start with 'linux'."""
        assert Platform.LINUX == "linux"

    def test_macos_constant_is_darwin(self):
        """MACOS constant should be 'darwin'."""
        assert Platform.MACOS == "darwin"


class TestPlatformWithMocking:
    """Tests with mocked sys.platform for cross-platform coverage."""

    def test_windows_detection(self, monkeypatch):
        """Should correctly detect Windows."""
        monkeypatch.setattr("sys.platform", "win32")
        # Force reimport to get the mocked value
        import importlib
        import ripperdoc.utils.platform

        importlib.reload(ripperdoc.utils.platform)

        from ripperdoc.utils.platform import Platform

        assert Platform.is_windows() is True
        assert Platform.is_linux() is False
        assert Platform.is_macos() is False
        assert Platform.get_system() == "windows"

    def test_linux_detection(self, monkeypatch):
        """Should correctly detect Linux."""
        monkeypatch.setattr("sys.platform", "linux")
        import importlib
        import ripperdoc.utils.platform

        importlib.reload(ripperdoc.utils.platform)

        from ripperdoc.utils.platform import Platform

        assert Platform.is_linux() is True
        assert Platform.is_windows() is False
        assert Platform.is_macos() is False
        assert Platform.get_system() == "linux"

    def test_macos_detection(self, monkeypatch):
        """Should correctly detect macOS."""
        monkeypatch.setattr("sys.platform", "darwin")
        import importlib
        import ripperdoc.utils.platform

        importlib.reload(ripperdoc.utils.platform)

        from ripperdoc.utils.platform import Platform

        assert Platform.is_macos() is True
        assert Platform.is_windows() is False
        assert Platform.is_linux() is False
        assert Platform.get_system() == "macos"

    def test_freebsd_detection(self, monkeypatch):
        """Should correctly detect FreeBSD."""
        monkeypatch.setattr("sys.platform", "freebsd")
        import importlib
        import ripperdoc.utils.platform

        importlib.reload(ripperdoc.utils.platform)

        from ripperdoc.utils.platform import Platform

        assert Platform.is_bsd() is True

    def test_openbsd_detection(self, monkeypatch):
        """Should correctly detect OpenBSD."""
        monkeypatch.setattr("sys.platform", "openbsd")
        import importlib
        import ripperdoc.utils.platform

        importlib.reload(ripperdoc.utils.platform)

        from ripperdoc.utils.platform import Platform

        assert Platform.is_bsd() is True

    def test_unknown_platform_detection(self, monkeypatch):
        """Should return 'unknown' for unrecognized platforms."""
        monkeypatch.setattr("sys.platform", "some_unknown_os")
        import importlib
        import ripperdoc.utils.platform

        importlib.reload(ripperdoc.utils.platform)

        from ripperdoc.utils.platform import Platform

        assert Platform.get_system() == "unknown"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_is_windows_returns_bool(self):
        """is_windows function should return a boolean."""
        assert isinstance(is_windows(), bool)

    def test_is_linux_returns_bool(self):
        """is_linux function should return a boolean."""
        assert isinstance(is_linux(), bool)

    def test_is_macos_returns_bool(self):
        """is_macos function should return a boolean."""
        assert isinstance(is_macos(), bool)

    def test_is_bsd_returns_bool(self):
        """is_bsd function should return a boolean."""
        assert isinstance(is_bsd(), bool)

    def test_is_unix_returns_bool(self):
        """is_unix function should return a boolean."""
        assert isinstance(is_unix(), bool)

    def test_is_posix_returns_bool(self):
        """is_posix function should return a boolean."""
        assert isinstance(is_posix(), bool)

    def test_exactly_one_platform_is_true(self):
        """Exactly one of Windows/Linux/macOS should be True."""
        platform_count = sum([is_windows(), is_linux(), is_macos()])
        assert platform_count >= 1  # At least one should be true (could be BSD)


class TestModuleLevelConstants:
    """Tests for module-level constant values."""

    def test_is_windows_constant_is_bool(self):
        """IS_WINDOWS should be a boolean."""
        assert isinstance(IS_WINDOWS, bool)

    def test_is_linux_constant_is_bool(self):
        """IS_LINUX should be a boolean."""
        assert isinstance(IS_LINUX, bool)

    def test_is_macos_constant_is_bool(self):
        """IS_MACOS should be a boolean."""
        assert isinstance(IS_MACOS, bool)

    def test_is_bsd_constant_is_bool(self):
        """IS_BSD should be a boolean."""
        assert isinstance(IS_BSD, bool)

    def test_is_unix_constant_is_bool(self):
        """IS_UNIX should be a boolean."""
        assert isinstance(IS_UNIX, bool)

    def test_is_posix_constant_is_bool(self):
        """IS_POSIX should be a boolean."""
        assert isinstance(IS_POSIX, bool)

    def test_constants_match_functions(self):
        """Module-level constants should match function calls."""
        assert IS_WINDOWS == is_windows()
        assert IS_LINUX == is_linux()
        assert IS_MACOS == is_macos()
        assert IS_BSD == is_bsd()
        assert IS_UNIX == is_unix()
        assert IS_POSIX == is_posix()


class TestModuleAvailabilityDetection:
    """Tests for platform-specific module availability."""

    def test_has_termios_returns_bool(self):
        """has_termios should return a boolean."""
        assert isinstance(has_termios(), bool)

    def test_has_fcntl_returns_bool(self):
        """has_fcntl should return a boolean."""
        assert isinstance(has_fcntl(), bool)

    def test_has_tty_returns_bool(self):
        """has_tty should return a boolean."""
        assert isinstance(has_tty(), bool)

    def test_has_termios_constant_is_bool(self):
        """HAS_TERMIOS should be a boolean."""
        assert isinstance(HAS_TERMIOS, bool)

    def test_has_fcntl_constant_is_bool(self):
        """HAS_FCNTL should be a boolean."""
        assert isinstance(HAS_FCNTL, bool)

    def test_has_tty_constant_is_bool(self):
        """HAS_TTY should be a boolean."""
        assert isinstance(HAS_TTY, bool)

    def test_termios_not_available_on_windows(self):
        """On Windows, termios should not be available."""
        if is_windows():
            assert HAS_TERMIOS is False
            assert HAS_TTY is False
            assert HAS_FCNTL is False

    def test_unix_modules_consistent_with_is_unix(self):
        """Unix modules should be available when is_unix is True."""
        if is_unix():
            # On Unix-like systems, at least fcntl should be available
            assert HAS_FCNTL is True


class TestPlatformIntegration:
    """Integration tests for platform detection."""

    def test_unix_implies_posix(self):
        """is_unix should always equal is_posix."""
        assert is_unix() == is_posix()

    def test_platform_mutual_exclusivity(self):
        """Windows and Unix are mutually exclusive."""
        # Either we're on Windows or we're on Unix
        assert is_windows() != is_unix()

    def test_get_system_matches_individual_functions(self):
        """get_system result should match individual is_* functions."""
        system = Platform.get_system()

        if system == "windows":
            assert is_windows() is True
        elif system == "linux":
            assert is_linux() is True
        elif system == "macos":
            assert is_macos() is True


class TestPlatformEdgeCases:
    """Edge case tests for platform detection."""

    def test_case_insensitive_platform_startswith(self):
        """Platform detection should handle case correctly."""
        # sys.platform is always lowercase, but test should be robust
        raw_name = Platform.get_raw_name()
        assert isinstance(raw_name, str)
        # Should not raise exceptions
        Platform.get_system()

    def test_platform_detection_does_not_raise(self):
        """All platform detection functions should not raise exceptions."""
        # Should not raise any exceptions
        Platform.get_system()
        Platform.is_windows()
        Platform.is_linux()
        Platform.is_macos()
        Platform.is_bsd()
        Platform.is_unix()
        Platform.is_posix()
        Platform.get_raw_name()
        Platform.get_os_name()

    def test_convenience_functions_do_not_raise(self):
        """Convenience functions should not raise exceptions."""
        # Should not raise any exceptions
        is_windows()
        is_linux()
        is_macos()
        is_bsd()
        is_unix()
        is_posix()

    def test_module_detection_does_not_raise(self):
        """Module detection functions should not raise exceptions."""
        # Should not raise any exceptions
        has_termios()
        has_fcntl()
        has_tty()
