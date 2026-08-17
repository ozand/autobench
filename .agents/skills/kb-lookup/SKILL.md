---
name: kb-lookup
description: Look up a fix in the local error knowledge base when a command fails, a tool errors, or behavior is unexpected. Matches the real error text against error_signatures in index.yaml and applies the matched lesson's resolution. Use whenever an error or failure occurs before retrying.
---

# kb-lookup — search for an error fix

Use this skill immediately when an error or failure occurs, BEFORE trying to guess the solution or retrying the command.

## How to use

1. Look for the local error index file (usually `.workspace-kb/index.yaml` or `kb/index.yaml`).
2. Search the file for keywords or exact substrings from the error you just received.
3. If an `error_signatures` block matches your error, note the `file` path pointing to the lesson.
4. Read the corresponding `KB-XXXX-*.md` file.
5. Apply the instructions found under the `## Resolution` heading of the lesson.

If no match is found, proceed with standard troubleshooting. Once resolved, consider using `kb-capture` to document the fix.
