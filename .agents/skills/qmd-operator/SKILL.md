---
name: qmd-operator
description: Manage local QMD semantic search, index collections, and execute semantic queries over raw documentation and system knowledge. Use when you need to search product documentation, update the semantic index after fetching new docs, or configure QMD collections.
---

# qmd-operator

Manage the local QMD (Semantic Search) workspace. This skill allows agents to configure QMD collections, update the vector index, and query the documentation.

## When to use

- You need to search for facts in product documentation.
- You just downloaded raw release notes or docs into `kb/raw/` (or `kb/apps/<app>/raw/`) and need to update the QMD index.
- You need to create a new QMD collection.

## Commands

### 1. Update the Index
After adding new files to the knowledge base:
```bash
qmd index update
```

### 2. Search and Query
To perform a fast keyword or semantic search across a specific collection:
```bash
qmd search "<keywords>" -c <collection_name>
```

For a complex semantic query specifying intent:
```bash
qmd query "intent: <what you are looking for>\nlex: <keywords>" -c <collection_name>
```

### 3. Read a Document
If QMD returns a reference to a file (e.g., `qmd://my-app/raw/releases.md`), use the context-mode tools to read the actual file from the disk.

## Configuring a New Collection
To define what files belong to a specific context, create a YAML file in `qmd/collections/<name>.yaml`:
```yaml
name: <collection_name>
description: "Knowledge base for <collection_name>"
paths:
  - "../../kb/"
exclude:
  - "**/.DS_Store"
```
Then run `qmd index update` to register it.
