---
name: kb-capture
description: Record a new lesson in the workspace error knowledge base after resolving a recurring or non-obvious error using OKF standard. Use after fixing an error that was not already in the KB and is likely to recur.
---

# kb-capture — record a new error lesson

Use this skill right after you resolve an error that was **not** already in the KB and is likely to recur or was non-obvious to diagnose.

## Steps

1. Find the local `index.yaml` (often in `.workspace-kb/index.yaml` or `kb/index.yaml`).
2. Determine the next free id (current max is `KB-XXXX`, use next).
3. Create a new markdown file `lessons/KB-XXXX-<slug>.md` with the OKF YAML Frontmatter.
4. Fill every required field. In particular:
   - `error_signatures` — **mandatory**. Use the most distinctive, stable parts of the real error (avoid timestamps, temp paths, PIDs).
5. Add an entry to the `index.yaml` with: `id`, `title`, `file`, `category`, `severity`, `tags`, `error_signatures`.

## OKF Error Markdown Template

```markdown
---
id: KB-XXXX
title: "<Clear title describing the failure>"
category: "<e.g., build, environment, runtime>"
severity: <low|medium|high>
tags: [tag1, tag2]
status: active
created: YYYY-MM-DD
updated: YYYY-MM-DD
error_signatures:
  - "<Stable part of error message>"
---

# <Title>

## Symptom
<What was observed>

## Root Cause
<Why it happened>

## Resolution
- Step 1
- Step 2

## Prevention
<How to avoid this in the future>
```
