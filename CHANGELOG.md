# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-02

### Added

- arXiv API integration: fetch paper abstracts by query terms, stored in SQLite
- Semantic Scholar API integration: citation graph retrieval (direct + co-citation edges)
- NV-Embed-v2 embedding pipeline with BAAI/bge-large-en-v1.5 as configurable fallback
- FAISS IndexFlatIP for efficient cosine similarity search over paper embeddings
- Dual-query retrieval: semantic query (researcher's field) + structural query (other domains)
- Cross-domain similarity search: finds paper pairs from different arXiv categories
- `arxanon search` CLI command with Rich terminal UI and progress tracking
- Configurable embedding model via `ARXANON_EMBED_MODEL` environment variable
- Configurable data directory via `ARXANON_DATA_DIR` environment variable
- SQLite schema supporting direct, co-citation, and bibliographic coupling edges (Phase 2)
