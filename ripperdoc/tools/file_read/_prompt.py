"""Read tool prompt."""

MAX_LINES_TO_READ = 2000

FILE_UNCHANGED_STUB = (
    "File unchanged since last read. The content from the earlier Read tool_result "
    "in this conversation is still current - refer to that instead of re-reading."
)

READ_PROMPT = """Read a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read valid file paths on the machine. If the User provides a path to a file, assume it is intended to be read; invalid paths return an error.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to """ + str(MAX_LINES_TO_READ) + """ lines starting from the beginning of the file
- You can optionally specify a line offset and limit for large files or focused reads
- Results are returned using cat -n format, with line numbers starting at 1
- This tool can read image files (eg PNG, JPG, etc) and returns an image placeholder with metadata rather than rendered visual content.
- This tool can read Jupyter notebooks (.ipynb files) and returns cell sources plus supported textual outputs.
- This tool can only read files, not directories. To read a directory, use the LS tool when available.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to inspect the file metadata/content available from the path.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."""
