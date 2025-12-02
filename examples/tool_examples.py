"""Example usage of Ripperdoc tools.

This file demonstrates how to use Ripperdoc programmatically.
"""

import asyncio
from ripperdoc.tools.bash_tool import BashTool, BashToolInput
from ripperdoc.tools.file_read_tool import FileReadTool, FileReadToolInput
from ripperdoc.tools.file_edit_tool import FileEditToolInput
from ripperdoc.tools.glob_tool import GlobTool, GlobToolInput
from ripperdoc.core.tool import ToolUseContext


async def example_bash_tool():
    """Example: Execute a bash command."""
    print("\n=== Bash Tool Example ===")

    tool = BashTool()
    context = ToolUseContext()

    input_data = BashToolInput(
        command="echo 'Hello from Ripperdoc!'",
        timeout=10000
    )

    async for result in tool.call(input_data, context):
        print(f"Command: {result.data.command}")
        print(f"Output: {result.data.stdout}")
        print(f"Exit Code: {result.data.exit_code}")


async def example_file_read_tool():
    """Example: Read a file."""
    print("\n=== File Read Tool Example ===")

    tool = FileReadTool()
    context = ToolUseContext()

    # Read this example file
    input_data = FileReadToolInput(
        file_path=__file__,
        limit=10  # Only read first 10 lines
    )

    async for result in tool.call(input_data, context):
        print(f"File: {result.data.file_path}")
        print(f"Lines read: {result.data.line_count}")
        print("Content preview:")
        print(result.data.content[:200] + "...")


async def example_glob_tool():
    """Example: Find Python files."""
    print("\n=== Glob Tool Example ===")

    tool = GlobTool()
    context = ToolUseContext()

    input_data = GlobToolInput(
        pattern="**/*.py",
        path="."
    )

    async for result in tool.call(input_data, context):
        print(f"Pattern: {result.data.pattern}")
        print(f"Found {result.data.count} files")
        if result.data.matches:
            print("First 5 matches:")
            for match in result.data.matches[:5]:
                print(f"  - {match}")


async def example_file_edit_tool():
    """Example: Edit a file (demo only, doesn't actually edit)."""
    print("\n=== File Edit Tool Example ===")
    print("Note: This is a demonstration. To actually edit files, use a real file.")

    # Show what input would look like
    print("\nExample input:")
    input_data = FileEditToolInput(
        file_path="/path/to/file.py",
        old_string="def old_function():",
        new_string="def new_function():"
    )
    print(f"  File: {input_data.file_path}")
    print(f"  Replace: '{input_data.old_string}'")
    print(f"  With: '{input_data.new_string}'")


async def example_query_workflow():
    """Example: Complete workflow with multiple tools."""
    print("\n=== Complete Workflow Example ===")

    context = ToolUseContext()

    # 1. Find Python files
    print("\n1. Finding Python files...")
    glob_tool = GlobTool()
    glob_input = GlobToolInput(pattern="*.py", path="ripperdoc/tools")

    async for result in glob_tool.call(glob_input, context):
        files = result.data.matches
        print(f"Found {len(files)} Python files in ripperdoc/tools")

    # 2. Read first file
    if files:
        print(f"\n2. Reading {files[0]}...")
        read_tool = FileReadTool()
        read_input = FileReadToolInput(file_path=files[0], limit=5)

        async for result in read_tool.call(read_input, context):
            print(f"First {result.data.line_count} lines:")
            print(result.data.content)

    # 3. List directory contents
    print("\n3. Listing directory contents...")
    bash_tool = BashTool()
    bash_input = BashToolInput(command="ls -la ripperdoc/tools")

    async for result in bash_tool.call(bash_input, context):
        print(result.data.stdout)


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Ripperdoc Tool Examples")
    print("=" * 60)

    await example_bash_tool()
    await example_file_read_tool()
    await example_glob_tool()
    await example_file_edit_tool()
    await example_query_workflow()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
