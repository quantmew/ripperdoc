#!/usr/bin/env python3
"""Test script for Git integration in LS tool."""

import asyncio
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ripperdoc.tools.ls_tool import LSTool, LSToolInput
from ripperdoc.core.tool import ToolUseContext
from ripperdoc.utils.git_utils import *


async def test_git_integration():
    """Test LS tool with Git repository."""
    print("Testing Git integration in LS tool...")
    
    # Create a temporary directory with a git repository
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, capture_output=True)
        
        # Create some files
        (tmp_path / "README.md").write_text("# Test Repository")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('Hello')")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test(): pass")
        
        # Create .gitignore
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n*.log\n")
        
        # Add and commit some files
        subprocess.run(["git", "add", "README.md", "src/"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, capture_output=True)
        
        # Create an untracked file
        (tmp_path / "untracked.txt").write_text("This is untracked")
        
        # Create a modified file
        (tmp_path / "src" / "main.py").write_text("print('Hello World')")
        
        print(f"\n1. Testing git utilities in: {tmp_path}")
        print(f"   Is git repository: {is_git_repository(tmp_path)}")
        print(f"   Git root: {get_git_root(tmp_path)}")
        print(f"   Current branch: {get_current_git_branch(tmp_path)}")
        print(f"   Commit hash: {get_git_commit_hash(tmp_path)}")
        print(f"   Working directory clean: {is_working_directory_clean(tmp_path)}")
        
        tracked, untracked = get_git_status_files(tmp_path)
        print(f"   Tracked files: {tracked}")
        print(f"   Untracked files: {untracked}")
        
        # Test gitignore patterns
        patterns = read_gitignore_patterns(tmp_path)
        print(f"\n2. Gitignore patterns: {patterns}")
        
        # Test LS tool
        print("\n3. Testing LS tool with Git integration...")
        tool = LSTool()
        input_data = LSToolInput(path=str(tmp_path))
        context = ToolUseContext()
        
        async for output in tool.call(input_data, context):
            if hasattr(output, 'data'):
                data = output.data
                print(f"   Root: {data.root}")
                print(f"   File count: {data.file_count}")
                print(f"   Git info: {data.git_info}")
                
                # Verify git info is present
                assert data.git_info, "Git info should be present"
                assert data.git_info.get("repository") == str(tmp_path)
                assert data.git_info.get("branch") == "main"
                assert data.git_info.get("commit") is not None
                assert data.git_info.get("clean") == "no (uncommitted changes)"
                assert "status" in data.git_info
                
                # Check that .git directory is not listed
                entries_str = "\n".join(data.entries)
                assert ".git/" not in entries_str, ".git directory should be ignored"
                
                # Check that gitignore patterns are respected
                # (we shouldn't see any .pyc files since they don't exist yet)
                print(f"   Entries: {data.entries}")
                
                print("\n✓ All Git integration tests passed!")
                return True
    
    return False


async def test_gitignore_parsing():
    """Test gitignore pattern parsing."""
    print("\n4. Testing gitignore pattern parsing...")
    
    test_patterns = [
        ("*.pyc", "*.pyc", None),
        ("__pycache__/", "__pycache__/", None),
        ("/dist", "dist", Path.cwd()),
        ("~/temp/*", "temp/*", Path.home()),
    ]
    
    for pattern, expected_rel, expected_root in test_patterns:
        rel, root = parse_gitignore_pattern(pattern, Path.cwd())
        print(f"   Pattern: {pattern} -> rel={rel}, root={root}")
        assert rel == expected_rel, f"Expected rel={expected_rel}, got {rel}"
        if expected_root is None:
            assert root is None, f"Expected root=None, got {root}"
        else:
            assert root == expected_root, f"Expected root={expected_root}, got {root}"
    
    print("✓ Gitignore parsing tests passed!")


async def test_ignore_map():
    """Test ignore patterns map building."""
    print("\n5. Testing ignore patterns map...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a simple .gitignore
        (tmp_path / ".gitignore").write_text("*.tmp\nbuild/\n")
        
        ignore_map = build_ignore_patterns_map(
            tmp_path,
            user_ignore_patterns=["*.log", "test_*.py"],
            include_gitignore=True
        )
        
        print(f"   Ignore map keys: {list(ignore_map.keys())}")
        
        # Should have entries for None (relative) and possibly git root
        assert None in ignore_map, "Should have relative patterns"
        patterns = ignore_map[None]
        print(f"   Relative patterns: {patterns}")
        
        # Should include both gitignore and user patterns
        assert "*.tmp" in patterns, "Should include gitignore pattern"
        assert "build/" in patterns, "Should include gitignore pattern"
        assert "*.log" in patterns, "Should include user pattern"
        assert "test_*.py" in patterns, "Should include user pattern"
    
    print("✓ Ignore map tests passed!")


async def main():
    """Run all tests."""
    try:
        await test_git_integration()
        await test_gitignore_parsing()
        await test_ignore_map()
        print("\n✅ All Git integration tests completed successfully!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)