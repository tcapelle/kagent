---
name: web-search-advanced-research-paper
description: Search for research papers and academic content using Exa advanced search. Full filter support including date ranges and domain filtering. Use when searching for academic papers, arXiv preprints, or scientific research.
---

# Web Search Advanced - Research Paper Category

Use the `exa_search` tool to search for research papers and academic content, and `web_fetch` to read the full text of the papers you pick.

## Tool Restriction (Critical)

ONLY use `exa_search` with `category: "publication"` for discovery. Use `web_fetch` only on URLs returned by a search.

## Filters

### Core
- `query` (required) — natural-language descriptions work well
- `objective` — what should rank first and what to extract (e.g. "papers with ablations on mesh attention for CFD surrogates")
- `numResults` (default 5, max 25)
- `type` (`"auto"`, `"lite"`, `"deep-lite"`, `"deep"`) — `deep` does multi-step research and is slower

### Domain filtering
- `includeDomains` (e.g. `["arxiv.org", "openreview.net"]`)
- `excludeDomains`

### Date filtering (ISO 8601)
- `startPublishedDate` / `endPublishedDate`

### Content extraction
- `contents` (`"highlights"` default, `"text"`, `"summary"`, `"none"`)
- `maxCharacters` — caps the returned text per result

## Context Budget (Critical)

Search results land in your own context. Keep them small:
- Prefer `contents: "highlights"` or `"summary"` for discovery; only use `"text"` on a handful of results
- Use `maxCharacters` when fetching full text
- Merge and deduplicate results, then write a distilled summary (brief markdown) before moving on

## When to Use

Use this category when you need:
- Academic papers from arXiv, OpenReview, PubMed, etc.
- Scientific research on specific topics
- Literature reviews with date filtering
- Papers on specific methodologies or terms

## Examples

Recent papers on a topic:
```
exa_search {
  "query": "transformer attention mechanisms efficiency",
  "category": "publication",
  "startPublishedDate": "2024-01-01",
  "numResults": 15,
  "type": "auto"
}
```

Papers from specific venues:
```
exa_search {
  "query": "neural surrogates for unstructured mesh CFD",
  "objective": "methods that beat Transolver on airfoil benchmarks, with reported numbers",
  "category": "publication",
  "includeDomains": ["arxiv.org", "openreview.net"],
  "numResults": 20,
  "type": "deep"
}
```

## Output Format

Return:
1) Results (structured list with title, authors, date, abstract summary)
2) Sources (URLs with publication venue)
3) Notes (methodology differences, conflicting findings)

## Follow-up

Use the initial summary findings to pick full papers to read with `web_fetch` (pass `query` to get only the relevant excerpts).
