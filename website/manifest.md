# Continuum Website Content Manifest

This document outlines the source of truth for the copy used in the generated product website, ensuring it maps accurately to the `/Users/mayanksahu/Continuum` codebase.

## 1. Hero Section
*   **Positional Statement:** "Intelligence with Continuity."
*   **Source Data:** `README.md` ("Continuum is an agent centric memory engine providing short-, mid- and long-term memory...")
*   **Status:** Grounded. Explains the core value proposition of preventing context collapse.

## 2. Core Capabilities (Features)
All features listed in this section are **Implemented (Available Now)** in the repository, with the exception of the API layer.
*   **Short-Term Memory (STM):** Confirmed via `memory/stm/ConversationSTM.py` (token-aware, compression logic).
*   **Mid-Term Memory (MTM):** Confirmed via `memory/mtm/repository/mtmRepository.py` (Postgres + pgvector, OpenAI embeddings).
*   **Live Knowledge Injection:** Confirmed via `continuum_mcp/web_search_server` and `helper/web_search.py`.
*   **API Layer:** Marked as **In Progress**. The `api` directory exists with a `main.py` stub, but business logic is not yet built out.

## 3. Architecture Pipeline
*   **Source Data:** `agent/AgentMTM.py` and `README.md` layout.
*   **Synthesis:** The diagram accurately represents the separation of Hot Memory (STM), Warm Memory (MTM), and the Orchestrator loop. 
*   **Status:** Grounded. Designed to demonstrate the modular interface-first approach (like the `STMCallbacks` injection).

## 4. Strategic Timeline (Roadmap)
*   **Phase 1 (Foundation - Hot & Warm Memory):** Marked as **Completed**. Matches the current CLI architecture stack.
*   **Phase 2 (Externalization - API Framework):** Marked as **In Progress**. Refers to the stubbed Fast API interface.
*   **Phase 3 (The Deep Archive - LTM):** Marked as **Planned**. LTM is mentioned in the `README` vision but no functional code for separate LTM exists yet.
*   **Phase 4 (Evolution - Advanced Scripts):** Marked as **Planned**. Synthesis of the repository's trajectory (moving beyond current naive summary techniques to graph mapping).

## Conclusion
The copy maintains a premium, aspirational "Memory OS" product aesthetic without hallucinating fake investors, users, or unbuilt logic. Where features are not fully coded (like the API or true LTM), they have been accurately badge-labeled as "Planned" or "In Progress."
