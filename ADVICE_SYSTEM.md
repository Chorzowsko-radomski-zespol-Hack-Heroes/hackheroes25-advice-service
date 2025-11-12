# Advice System Overview

## High-Level Flow
1. **Request intake (`/advice`)**
   - Accepts `user_id` or `Authorization` token and the latest user message (`message` query param).
   - Logs request metadata, validates presence of an identifier, and delegates to `AdviceService`.

2. **AdviceService**
   - Composes an `AdviceSelectionPipeline` with repositories, classifiers, and response generator.
   - For Supabase mode, builds:
     - `SupabaseAdviceRepository`
     - `SupabaseAdviceCategoryRepository`
     - `OpenAIEmbeddingCategoryClassifier`
     - `TrivialAdviceIntentDetector`
     - `EchoAdviceResponseGenerator` (placeholder)

3. **AdviceSelectionPipeline**
   - **Category inference**
     - `OpenAIEmbeddingCategoryClassifier` embeds the message using `text-embedding-3-large`.
     - Produces the top 6 category matches (`CategoryMatch` objects) containing the OpenAI similarity score and rank (position in descending order).
     - Matches names against categories stored in Supabase (`advice_categories.name`) using a flexible variant map (supports lowercase, ASCII, `-`, `_`).
   - **Intent detection**
     - Currently trivial (always returns `None`). Hook available for future enhancements.
   - **Candidate retrieval**
     - If preferred kind is present, fetches advices of that kind filtered by matched categories, otherwise by overlap, falling back to the entire catalogue.
   - **Ranking & selection**
     - Category usage frequencies are lazily cached (one pass on all advices). Frequencies help identify rare categories.
     - For each candidate the pipeline:
       - Computes a specificity factor based on the number of categories attached to the advice (fewer categories ⇒ higher weight).
       - Aggregates contributions from every overlapping `CategoryMatch`:
         - `similarity_weight` × `ranking_weight` × `rarity_weight` × `specificity_factor`.
         - `ranking_weight` leverages the classifier order (1…N).
         - `rarity_weight` uses `(total_advices + 1)/(frequency + 1)`. Extremely rare categories (frequency = 1) receive a strong boost, especially if they appear in TOP2.
       - Adds jitter (±10%) to avoid deterministic behaviour.
     - If a candidate is the sole advice containing the top-ranked category (frequency = 1 and rank = 1) it is returned immediately (100% probability). Rank 2 still receives a very high boost.
     - Otherwise applies a weighted random choice across all candidates based on the computed weights, ensuring a mix of determinism and variety.
   - **Response rendering**
     - `AdviceRecommendation` wraps the advice domain object and a placeholder chat response (to be replaced by LLM-driven text).

4. **Response serialization**
   - `AdviceResponsePayload` (Pydantic) exposes the advice details and chat response.

## Data Layer
### Advice Repository (`SupabaseAdviceRepository`)
- Retrieves full advice records with joined categories (`advice_category_links → advice_categories.name`).
- Supports:
  - `get_all()`
  - `get_by_kind(kind)`
  - `get_by_kind_and_containing_any_category(kind, categories)` – filters by advice kind and any of the provided category names.
- Returns domain `Advice` objects with category names exactly as stored in Supabase.

### Category Repository (`SupabaseAdviceCategoryRepository`)
- Returns the list of category names (`advice_categories.name`) ordered alphabetically.
- Simple membership checks reuse the cached list instead of additional queries.

## Classifiers & Scoring Rules
### Category Matches
- `CategoryMatch` has `name`, `score` (embedding cosine similarity), and `rank` (1 = most relevant).
- Pipeline considers TOP 6 matches and logs all scores.

### Rarity & Ranking Influence
- Category rarity: advices that are the unique holder of a HIGH-ranked category are prioritised (rank=1 ⇒ deterministic win, rank=2 ⇒ very high weight).
- Rank weight: `(len(matches) - rank + 1) / len(matches)`; higher-ranked categories contribute more.
- Specificity factor: `(max_item_categories + 1) / (len(advice.categories) + 1)` rewards narrower advices without penalising broad ones excessively.
- Random jitter ensures repeated identical requests can yield slightly different valid answers.

## Configuration & Limits
- **Max classifier categories**: 6 (TOP6 considered in scoring).
- **Max categories per advice**: pipeline assumes rare cases up to 7 (configurable via `max_item_categories`).
- **OpenAI settings**: read from environment (`OPENAI_API_KEY`, optional `OPENAI_ORGANIZATION`, `OPENAI_PROJECT`, `OPENAI_EMBEDDINGS_MODEL`).
- **Supabase settings**: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`.

## Extensibility Notes
- Intent detection is a plug-in point for future models (e.g. detecting explicit requests for `AdviceKind`).
- Response generation currently echoes placeholders – replace with an LLM-backed generator for production.
- Category frequencies cache is computed on first use; invalidate manually if the catalogue changes frequently (e.g., inject a repository signal).
- Weighted selection can be tuned by adjusting rarity multipliers, jitter amplitude, or specificity formula.

## Operational Checklist
- Ensure Supabase tables:
  - `advices` with category links
  - `advice_categories` containing human-readable `name`
  - `advice_category_links` mapping advices ↔ categories
- Populate `OPENAI_CATEGORY_DEFINITIONS` in `advice_service.py` to stay in sync with Supabase categories.
- Use `uvicorn app.main:app --reload --env-file .env` for local runs; `.env` should contain Supabase and OpenAI credentials.

