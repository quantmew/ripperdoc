"""File mention completer for @ symbol completion.

This module provides file path completion when users type @ followed by a filename.
Supports recursive search across the entire project.
"""

from pathlib import Path
from typing import Any, Iterable, List, Set

from prompt_toolkit.completion import Completer, Completion

from ripperdoc.utils.path_ignore import should_skip_path, IgnoreFilter


class FileMentionCompleter(Completer):
    """Autocomplete for file paths when typing @.

    Supports recursive search - typing 'cli' will match 'ripperdoc/cli/cli.py'
    """

    def __init__(self, project_path: Path, ignore_filter: IgnoreFilter):
        """Initialize the file mention completer.

        Args:
            project_path: Root path of the project
            ignore_filter: Pre-built ignore filter for filtering files
        """
        self.project_path = project_path
        self.ignore_filter = ignore_filter

    def _collect_files_recursive(self, root_dir: Path, max_depth: int = 5) -> List[Path]:
        """Recursively collect all files from root_dir, respecting ignore rules.

        Args:
            root_dir: Directory to search from
            max_depth: Maximum directory depth to search

        Returns:
            List of file paths relative to project root
        """
        files = []

        def _walk(current_dir: Path, depth: int) -> None:
            if depth > max_depth:
                return

            try:
                for item in current_dir.iterdir():
                    # Use the project's ignore filter to skip files
                    if should_skip_path(
                        item,
                        self.project_path,
                        ignore_filter=self.ignore_filter,
                        skip_hidden=True,
                    ):
                        continue

                    if item.is_file():
                        files.append(item)
                    elif item.is_dir():
                        # Recurse into subdirectory
                        _walk(item, depth + 1)
            except (OSError, PermissionError):
                # Skip directories we can't read
                pass

        _walk(root_dir, 0)
        return files

    def get_completions(self, document: Any, complete_event: Any) -> Iterable[Completion]:
        """Get completion suggestions for the current input.

        Args:
            document: The current document/input
            complete_event: Completion event

        Yields:
            Completion objects with file paths
        """
        text = document.text_before_cursor

        # Find the last @ symbol in the text
        at_pos = text.rfind("@")
        if at_pos == -1:
            return

        # Extract the query after the @ symbol
        query = text[at_pos + 1 :].strip()

        try:
            matches = []
            seen: Set[str] = set()

            def _add_match(display_path: str, item: Path, meta: str, score: int) -> None:
                if display_path in seen:
                    return
                seen.add(display_path)
                matches.append((display_path, item, meta, score))

            # If query contains path separator, do directory-based search
            if "/" in query or "\\" in query:
                # User is typing a specific path
                query_path = Path(query.replace("\\", "/"))

                if query.endswith(("/", "\\")):
                    # Show contents of this directory
                    search_dir = self.project_path / query_path
                    if search_dir.exists() and search_dir.is_dir():
                        for item in sorted(search_dir.iterdir()):
                            if should_skip_path(
                                item,
                                self.project_path,
                                ignore_filter=self.ignore_filter,
                                skip_hidden=True,
                            ):
                                continue

                            try:
                                rel_path = item.relative_to(self.project_path)
                                display_path = str(rel_path)
                                if item.is_dir():
                                    display_path += "/"

                                # Right side: show type only
                                meta = "📁 directory" if item.is_dir() else "📄 file"

                                _add_match(display_path, item, meta, 0)
                            except ValueError:
                                continue
                else:
                    # Match files in the parent directory
                    parent_dir = self.project_path / query_path.parent
                    pattern = f"{query_path.name}*"

                    if parent_dir.exists() and parent_dir.is_dir():
                        for item in sorted(parent_dir.iterdir()):
                            if should_skip_path(
                                item,
                                self.project_path,
                                ignore_filter=self.ignore_filter,
                                skip_hidden=True,
                            ):
                                continue

                            import fnmatch

                            if fnmatch.fnmatch(item.name.lower(), pattern.lower()):
                                try:
                                    rel_path = item.relative_to(self.project_path)
                                    display_path = str(rel_path)
                                    if item.is_dir():
                                        display_path += "/"

                                    # Right side: show type only
                                    meta = "📁 directory" if item.is_dir() else "📄 file"

                                    _add_match(display_path, item, meta, 0)
                                except ValueError:
                                    continue
            else:
                # Recursive search: match query against filename anywhere in project
                if not query:
                    # No query: show top-level items only
                    for item in sorted(self.project_path.iterdir()):
                        if should_skip_path(
                            item,
                            self.project_path,
                            ignore_filter=self.ignore_filter,
                            skip_hidden=True,
                        ):
                            continue

                        try:
                            rel_path = item.relative_to(self.project_path)
                            display_path = str(rel_path)
                            if item.is_dir():
                                display_path += "/"

                            # Right side: show type only
                            meta = "📁 directory" if item.is_dir() else "📄 file"
                            _add_match(display_path, item, meta, 0)
                        except ValueError:
                            continue
                else:
                    # First, suggest top-level entries that match the prefix to support step-by-step navigation
                    query_lower = query.lower()
                    for item in sorted(self.project_path.iterdir()):
                        if should_skip_path(
                            item,
                            self.project_path,
                            ignore_filter=self.ignore_filter,
                            skip_hidden=True,
                        ):
                            continue

                        name_lower = item.name.lower()
                        if query_lower in name_lower:
                            score = 500
                            if name_lower.startswith(query_lower):
                                score += 50
                            if name_lower == query_lower:
                                score += 100

                            rel_path = item.relative_to(self.project_path)
                            display_path = str(rel_path)
                            if item.is_dir():
                                display_path += "/"

                            meta = "📁 directory" if item.is_dir() else "📄 file"
                            _add_match(display_path, item, meta, score)

                    # If the query exactly matches a directory, also surface its children for quicker drilling
                    dir_candidate = self.project_path / query
                    if dir_candidate.exists() and dir_candidate.is_dir():
                        for item in sorted(dir_candidate.iterdir()):
                            if should_skip_path(
                                item,
                                self.project_path,
                                ignore_filter=self.ignore_filter,
                                skip_hidden=True,
                            ):
                                continue

                            try:
                                rel_path = item.relative_to(self.project_path)
                                display_path = str(rel_path)
                                if item.is_dir():
                                    display_path += "/"
                                meta = "📁 directory" if item.is_dir() else "📄 file"
                                _add_match(display_path, item, meta, 400)
                            except ValueError:
                                continue

                    # Recursively search for files matching the query
                    all_files = self._collect_files_recursive(self.project_path)

                    for file_path in all_files:
                        try:
                            rel_path = file_path.relative_to(self.project_path)
                            file_name = file_path.name

                            # Match against filename
                            if query_lower in file_name.lower():
                                # Calculate relevance score (prefer exact matches and shorter names)
                                score = 0
                                if file_name.lower().startswith(query_lower):
                                    score += 100  # Prefix match is highly relevant
                                if file_name.lower() == query_lower:
                                    score += 200  # Exact match is most relevant
                                score -= len(str(rel_path))  # Prefer shorter paths

                                display_path = str(rel_path)

                                # Right side: show type only
                                meta = "📄 file"

                                _add_match(display_path, file_path, meta, score)
                        except ValueError:
                            continue

            # Sort matches by score (descending) and then by path
            matches.sort(key=lambda x: (-x[3], x[0]))

            # Limit results to prevent overwhelming the user
            matches = matches[:50]

            for display_path, item, meta, score in matches:
                yield Completion(
                    display_path,
                    start_position=-(len(query) + 1),  # +1 to include the @ symbol
                    display=display_path,
                    display_meta=meta,
                )

        except (OSError, ValueError, RuntimeError):
            # Silently ignore errors during completion
            pass
