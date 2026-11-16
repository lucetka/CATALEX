# 🐈 CATALEX

**Cosine Alignment of Topics via OpenAlex**

CATALEX is a lightweight Streamlit dashboard for mapping journal scope and finding structurally similar journals from live OpenAlex primary-topic profiles.

CATALEX represents each journal as a vector of topic shares (currently using exclusively the PrimaryTopic from the OpenAlex taxonomy). It then uses cosine similarity, or an optional TF-IDF-weighted variant, to identify journals with a comparable publication footprint.

## What CATALEX does

- Resolves a target journal against the OpenAlex Sources endpoint.
- Builds its recent primary-topic distribution.
- Finds journals publishing in the target's core topics.
- Fetches complete candidate topic profiles concurrently.
- Ranks candidates by cosine similarity or TF-IDF cosine similarity.
- Displays the target and candidates in a heatmap, radar chart, and ranking table.
- Generates a Boolean journal query from the top-ranked candidates.
- Compares two time windows and measures scope drift.
- Offers **Chasing Mode**, which ranks journals against the target's positive topic-share changes rather than its standard profile.

## Why it exists

Finding meaningful competitor journals is difficult when broad indexing categories place highly multidisciplinary and tightly focused journals together. CATALEX instead compares each journal's observed global publication footprint in the OpenAlex taxonomy.

This helps specialty journals rise in the ranking when their topic mixture genuinely resembles the target, while broad journals are less likely to rank highly simply because they publish in many areas.

## Main controls

| Control | What it changes |
|---|---|
| Target Journal Name | Journal to resolve and analyse. |
| Filter Works From Date | Start date for the standard profile, in `YYYY-MM-DD` format. |
| Similarity Metric | Raw cosine similarity or TF-IDF-weighted cosine. |
| Max Journals Fetched per Topic | Breadth of candidate discovery for each core topic. |
| Minimum Papers per Shared Topic | Baseline volume required in a shared topic. CATALEX may raise this dynamically for large topics. |
| Minimum Similarity Threshold | Minimum final alignment score, from 0 to 1. |
| Max Candidate Profile Scans Ceiling | Safety ceiling for full candidate-profile API calls. |
| Whitelist Journals | Existing preset titles allowed through the similarity threshold if they first enter the discovered candidate pool. |
| Max Competitors Shown | Maximum number displayed after ranking and filtering. |

## Quick start

### 1. Install dependencies

```bash
pip install streamlit requests pandas numpy plotly
```

### 2. Run the app

```bash
streamlit run CATALEX.py
```

### 3. Configure a standard comparison

1. Enter the journal title.
2. Enter a start date such as `2023-01-01`.
3. Start with **Cosine Similarity** and the default thresholds.
4. Review the resolved OpenAlex title and source ID before interpreting results.
5. Use the ranking table, heatmap, and radar chart together. A high score is most useful when supported by a sensible shared-topic pattern and adequate publication volume.

## Temporal evolution and Chasing Mode

Open **Analyze Journal Scope Evolution in Time** and enter two inclusive year windows, for example:

- Epoch 1: `2016-2020`
- Epoch 2: `2021-2026`

CATALEX calculates:

- **Scope drift** as `1 - cosine similarity` between the two complete topic-share vectors.
- Topic-share changes in percentage points.
- A normalized positive-growth vector containing topics whose share increased.

When **Chasing Mode** is activated, candidate discovery, similarity ranking, the core-topic table, and the visual diagnostics all use that growth vector. This answers a different question from standard mode:

> Which journals currently resemble the areas in which the target journal is expanding?

The drift label is descriptive rather than a validated editorial benchmark. The current `0.15` boundary should be treated as a practical heuristic, not a universal definition of scope change.

## Interpreting the similarity metrics

### Cosine Similarity

Compares the shape of journal topic-share vectors. It is insensitive to total journal size, so two journals can align strongly even if one publishes many more papers.

### TF-IDF Cosine

Down-weights topics present across many journals in the current candidate set. This can highlight more distinctive shared areas, but the score depends on which candidates were discovered and scanned in that run. It is therefore less directly comparable across runs with different settings.

## Important limitations

1. **Journal resolution is still permissive.** If no exact case-insensitive title match is found among the first search results, the app accepts the first OpenAlex result. Always check the displayed resolved title and ID.
2. **Whitelist is not yet a true forced lookup.** A preset journal bypasses later filtering only if it was already found during topic-based candidate discovery. Free journal search and guaranteed injection still need implementation.
3. **OpenAlex data quality applies.** Source assignment, primary-topic classification, and historical coverage may contain errors or change over time.
4. **Candidate discovery is intentionally truncated.** Topic retrieval depth and the candidate scan ceiling trade completeness for speed and API cost.
5. **The dynamic topic-volume rule can be restrictive.** CATALEX requires at least 10% of the target journal's count in a topic, subject to the user minimum. This favours substantial competitors and may exclude small emerging journals.
6. **TF-IDF is run-specific.** IDF weights are calculated from the target plus the candidate profiles loaded in the current run, not from all OpenAlex journals.
7. **API credentials participate in Streamlit cache keys.** The password widget hides the key in the interface, but deployment and log handling should still follow normal secret-management practice.
8. **The query generator does not escape quotation marks inside journal titles.** This is rare, but generated queries should be checked before reuse.

## Reliability improvements in this revision

- Validates standard dates and temporal year windows.
- Converts temporal windows to explicit `from_publication_date` and `to_publication_date` filters.
- Retries transient OpenAlex `429` and `5xx` responses with bounded backoff.
- Prevents request-parameter dictionaries from being mutated.
- Uses the same active vector for candidate discovery, scoring, and charts in Chasing Mode.
- Reports how many candidate profiles failed instead of silently treating all failures as genuine zero similarity.
- Adds explanatory comments around the main retrieval and vector logic.

## Planned Features and Fixes 🚀

### Near term

- Replace the fixed whitelist options with OpenAlex-backed journal search and store selected OpenAlex source IDs rather than display names.
- Add a confirmation step when target resolution is not exact.
- Add CSV export for ranking and topic-profile tables, plus HTML download buttons for Plotly figures.
- Add optional uploaded journal metrics and filtering. Keep licensed internal metrics, such as Journal Impact Factor data, out of a public deployment.
- Add a compact diagnostics panel showing API failures, effective thresholds, candidate counts at each filtering stage, and the active OpenAlex source IDs.
- Add automated tests for date parsing, cosine calculations, candidate truncation, and Chasing Mode topic selection.
- Add higher granularity by harnessing other parts of OpenAlex curated taxonomy (especially keywords;  integrate with LucKeyWords https://luckeywords.streamlit.app/ or implement some of its features

### Longer term

- Compare journal scope across multiple rolling periods.
- Compare researchers and institutions using equivalent normalized topic vectors.
- Add vector-space taxonomy maps using stable topic embeddings and dimensionality reduction.
- Revisit the visual intuition of the 2023 **FIASCO** prototype: contour or density maps over a shared scientific landscape, without returning to a brittle manually refreshed overlap matrix.

## FIASCO background

FIASCO was the 2023 predecessor based on a precomputed overlap matrix and two-dimensional density maps produced from SPECTER embeddings reduced with UMAP. CATALEX replaces that static architecture with live OpenAlex topic profiles. A future embedding view could restore the spatial intuition while keeping the data-refresh workflow manageable.

## Responsible use

CATALEX is a quick screening and discovery tool, not an authoritative classification system. Competitor lists should be checked against journal aims and scope, editorial strategy, language, article types, publication model, and any licensed internal metrics before decisions are made.
