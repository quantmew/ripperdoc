"""Bash parser constants and language loading.

Provides the tree-sitter bash grammar singleton and SHELL_KEYWORDS constant.
"""

from __future__ import annotations

from typing import Any, Optional

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Shell keywords that can appear as the first word of a simple command.
# These are NOT commands themselves but bash syntax elements.
SHELL_KEYWORDS: frozenset[str] = frozenset({
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "case",
    "esac",
    "for",
    "while",
    "until",
    "do",
    "done",
    "in",
    "function",
    "select",
    "time",
    "coproc",
    "[[",
    "]]",
    "!",
    "{",
    "}",
})

_bash_language: Any = None
_bash_parser: Any = None


def get_bash_language() -> Any:
    """Load and cache the tree-sitter bash language.

    Returns:
        The tree-sitter Language object for bash.

    Raises:
        ImportError: If tree-sitter-bash is not installed.
    """
    global _bash_language
    if _bash_language is not None:
        return _bash_language

    try:
        from tree_sitter import Language
        from tree_sitter_bash import language as bash_language_fn

        capsule = bash_language_fn()
        _bash_language = Language(capsule)
        return _bash_language
    except ImportError:
        logger.warning(
            "tree-sitter-bash not installed. "
            "Install with: pip install tree-sitter-bash"
        )
        raise
    except Exception as exc:
        logger.warning(
            "[bash_parser] Failed to load bash language: %s: %s",
            type(exc).__name__, exc,
        )
        raise


def get_bash_parser() -> Any:
    """Get or create a cached tree-sitter Parser for bash.

    Returns:
        A tree-sitter Parser configured with the bash language.

    Raises:
        ImportError: If tree-sitter or tree-sitter-bash is not installed.
    """
    global _bash_parser
    if _bash_parser is not None:
        return _bash_parser

    try:
        import tree_sitter as ts

        lang = get_bash_language()
        _bash_parser = ts.Parser(lang)
        return _bash_parser
    except TypeError:
        # Older tree-sitter API: Parser(lang) may fail with newer bindings
        try:
            import tree_sitter as ts
            _bash_parser = ts.Parser()
            _bash_parser.language = get_bash_language()
            return _bash_parser
        except Exception as exc2:
            logger.warning(
                "[bash_parser] Failed to create parser with fallback: %s: %s",
                type(exc2).__name__, exc2,
            )
            raise
    except ImportError:
        logger.warning(
            "tree-sitter not installed. Install with: pip install tree-sitter"
        )
        raise
    except Exception as exc:
        logger.warning(
            "[bash_parser] Failed to create parser: %s: %s",
            type(exc).__name__, exc,
        )
        raise
