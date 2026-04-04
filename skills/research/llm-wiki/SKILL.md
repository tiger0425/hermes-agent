---
name: llm-wiki
description: "Karpathy's LLM Wiki — build and maintain a persistent, interlinked markdown knowledge base. Ingest sources, query compiled knowledge, and lint for consistency."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [wiki, knowledge-base, research, notes, markdown, rag-alternative]
    category: research
    related_skills: [obsidian, arxiv, agentic-research-ideas]
---

# Karpathy's LLM Wiki

Build and maintain a persistent, compounding knowledge base as interlinked markdown files.
Based on [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

Unlike traditional RAG (which rediscovers knowledge from scratch per query), the wiki
compiles knowledge once and keeps it current. Cross-references are already there.
Contradictions have already been flagged. Synthesis reflects everything ingested.

**Division of labor:** The human curates sources and directs analysis. The agent
summarizes, cross-references, files, and maintains consistency.

## Wiki Location

Set `LLM_WIKI_PATH` in `~/.hermes/.env` to choose where the wiki lives.

Default: `~/wiki`

The wiki is just a directory of markdown files — open it in Obsidian, VS Code, or
any editor. No database, no special tooling required.

## Architecture: Three Layers

```
wiki/
├── SCHEMA.md           # Conventions, structure rules, domain config
├── index.md            # Content catalog — every page with one-line summary
├── log.md              # Chronological action log (append-only)
├── raw/                # Layer 1: Immutable source material
│   ├── articles/       # Web articles, clippings
│   ├── papers/         # PDFs, arxiv papers
│   ├── transcripts/    # Meeting notes, interviews
│   └── assets/         # Images, diagrams referenced by sources
├── entities/           # Layer 2: Entity pages (people, orgs, products)
├── concepts/           # Layer 2: Concept/topic pages
├── comparisons/        # Layer 2: Side-by-side analyses
└── queries/            # Layer 2: Filed query results worth keeping
```

**Layer 1 — Raw Sources:** Immutable. The agent reads but never modifies these.
**Layer 2 — The Wiki:** Agent-owned markdown files. Created, updated, and
cross-referenced by the agent.
**Layer 3 — The Schema:** `SCHEMA.md` defines structure and conventions.

## Initializing a Wiki

When the user asks to create or start a wiki:

1. Determine the wiki path (from `LLM_WIKI_PATH` env var, or ask the user, or default to `~/wiki`)
2. Create the directory structure above
3. Write `SCHEMA.md` customized to the user's domain (ask what the wiki is about)
4. Write initial `index.md` with header and empty catalog
5. Write initial `log.md` with creation entry
6. Confirm the wiki is ready

```bash
# Check/create wiki path
WIKI="${LLM_WIKI_PATH:-$HOME/wiki}"
```

### SCHEMA.md Template

Adapt this to the user's domain. The schema prevents generic chatbot behavior
and makes the agent a disciplined maintainer:

```markdown
# Wiki Schema

## Domain
[What this wiki covers — e.g., "AI/ML research", "personal health", "startup intelligence"]

## Conventions
- File names: lowercase, hyphens, no spaces (e.g., `transformer-architecture.md`)
- Every wiki page starts with a YAML frontmatter block:
  ```yaml
  ---
  title: Page Title
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  type: entity | concept | comparison | query | summary
  tags: [tag1, tag2]
  sources: [raw/articles/source-name.md]
  ---
  ```
- Use `[[wikilinks]]` to link between pages
- When updating a page, always bump the `updated` date
- Every new page must be added to `index.md`
- Every action must be appended to `log.md`

## Entity Pages
One page per notable entity (person, company, model, tool). Include:
- Overview / what it is
- Key facts and dates
- Relationships to other entities
- Source references

## Concept Pages
One page per concept or topic. Include:
- Definition / explanation
- Current state of knowledge
- Open questions or debates
- Related concepts (wikilinks)

## Comparison Pages
Side-by-side analyses. Include:
- What is being compared and why
- Dimensions of comparison (table format preferred)
- Verdict or synthesis
- Sources

## Source Summaries
When ingesting a raw source, write a summary in the source's directory:
`raw/articles/source-name.md` (the original) gets a companion summary that
lives as a wiki page.
```

### index.md Template

```markdown
# Wiki Index

> Content catalog. Every wiki page with a one-line summary.
> Read this first to find relevant files for any query.

## Entities
<!-- entity pages listed here -->

## Concepts
<!-- concept pages listed here -->

## Comparisons
<!-- comparison pages listed here -->

## Queries
<!-- filed query results listed here -->
```

### log.md Template

```markdown
# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, delete

## [YYYY-MM-DD] create | Wiki initialized
- Domain: [domain]
- Structure created with SCHEMA.md, index.md, log.md
```

## Core Operations

### 1. Ingest

When the user provides a source (URL, file, paste), integrate it into the wiki:

① **Capture the raw source:**
   - URL → use `web_extract` to get markdown, save to `raw/articles/`
   - PDF → use `web_extract` (handles PDFs), save to `raw/papers/`
   - Pasted text → save to appropriate `raw/` subdirectory
   - Name the file descriptively: `raw/articles/karpathy-llm-wiki-2026.md`

② **Discuss takeaways** with the user — what's interesting, what matters for the domain.

③ **Write or update wiki pages:**
   - Create a summary page if the source is substantial
   - Create or update entity pages for key people/orgs/tools mentioned
   - Create or update concept pages for key ideas
   - Add cross-references (`[[wikilinks]]`) between new and existing pages

④ **Update navigation:**
   - Add new pages to `index.md` with one-line summaries
   - Append to `log.md`: `## [YYYY-MM-DD] ingest | Source Title`

⑤ **Report what changed** — list every file created or updated.

A single source can trigger updates across 10-15 wiki pages. This is normal
and desired — it's the compounding effect.

### 2. Query

When the user asks a question:

① **Read `index.md`** to identify relevant pages.
② **Read the relevant pages** using `read_file`.
③ **Synthesize an answer** from the compiled knowledge.
④ **File valuable answers back** — if the answer is a substantial comparison,
   deep dive, or discovery, create a new page in `queries/` or `comparisons/`
   so it doesn't disappear into chat history.
⑤ **Update log.md** with the query.

For large wikis (100+ pages), use `search_files` to search content across all
markdown files before synthesizing.

### 3. Lint

When the user asks to lint, health-check, or audit the wiki:

① **Scan for contradictions:** Read pages on the same topic and flag conflicting claims.
② **Find orphan pages:** Pages with no inbound `[[wikilinks]]` from other pages.
③ **Check for stale content:** Pages whose `updated` date is old relative to newer sources.
④ **Identify data gaps:** Topics referenced but lacking dedicated pages.
⑤ **Verify index completeness:** Every wiki page should appear in `index.md`.
⑥ **Report findings** with specific file paths and suggested actions.
⑦ **Append to log.md:** `## [YYYY-MM-DD] lint | N issues found`

Lint command for finding orphan pages:

```python
# Find all wiki pages
# For each, check if any OTHER page links to it via [[filename]]
# Pages with zero inbound links are orphans
```

## Working with the Wiki

### Searching

```bash
# Find pages by content
search_files "transformer" path="$WIKI" file_glob="*.md"

# Find pages by filename
search_files "*.md" target="files" path="$WIKI"

# Recent log activity
grep "^## \[" "$WIKI/log.md" | tail -10
```

### Bulk Operations

When ingesting multiple sources at once, batch the updates:
- Read all sources first
- Identify all entities and concepts across sources
- Create/update pages in one pass (avoids redundant updates)
- Update index.md once at the end
- Write a single log entry covering the batch

### Obsidian Integration

If the user has Obsidian, the wiki directory works as a vault out of the box:
- `[[wikilinks]]` render as clickable links
- Graph View visualizes the knowledge network
- YAML frontmatter powers Dataview queries
- The `raw/assets/` folder holds images referenced via `![[image.png]]`

Set `OBSIDIAN_VAULT_PATH` to the same directory as `LLM_WIKI_PATH` to use
both the Obsidian skill and this skill on the same vault.

## Pitfalls

- **Never modify files in `raw/`** — sources are immutable. Corrections go in wiki pages.
- **Always update index.md and log.md** — skipping this makes the wiki degrade over time.
- **Don't create pages without cross-references** — isolated pages are invisible. Every
  page should link to at least one other page.
- **Frontmatter is required** — it enables search, filtering, and staleness detection.
- **Keep summaries concise** — a wiki page should be scannable in 30 seconds. Move
  detailed analysis to dedicated deep-dive pages.
- **Ask before mass-updating** — if an ingest would touch 10+ existing pages, confirm
  the scope with the user first.
