🐈CATALEX 

Cosine Alignment of Topics via OpenAlex
A dynamic, lightweight Streamlit dashboard that maps and matches journal scopes using real-time topic footprints from the OpenAlex taxonomy.

Why this exists

Finding true "competitor" journals (journals with a similar scope) usually means wrestling with multi-disciplinary noise or rigid, static indexing. This tool uses vector geometry (cosine similarity) to align a target journal's specific primary-topic fingerprint against candidates globally.
Instead of treating multi-disciplinary mega-journals and razor-focused niche journals the same, it evaluates similarity against their true global publication footprint—guaranteeing that dedicated specialty competitors rise to the top while generic titles drop out.

Key Features

• 🧬 True Scope Matching: Computes cosine similarity based on actual global field distributions.
• 🎛️ Tunable Parameters: Live sliders for extraction depth, minimum topic density thresholds, and dynamic similarity cuts.
• 📌 Smart Whitelisting: Pins specific core titles to bypass threshold filters for guaranteed structural comparisons (i.e., if you want to force a comparison to a journal /  group of specific journals).
• 📊 Visual Diagnostics: Includes an interactive 2D Scope Overlay Heatmap and symmetry-mapped Radar Footprint overlapping charts (via Plotly).
• ⚡ Optimized Memory Matrix: Eliminates sequential journal API request loops by utilizing pre-aggregated topic relationship profiles.


🚀 Planned Features & Architectural Horizons

1. Robust Target Disambiguation & Semantic Matching

• The Upgrade: Implement an optional smart resolution pipeline using the OpenAlex autocomplete or search endpoint for target journals.
• Why it matters: While exact matching works perfectly for clear single-word titles like PROTEOMICS, it easily breaks down on complex or punctuated strings like Biochimica et Biophysica Acta - Proteins & Proteomics. Adding a fuzzy search stage will allow the app to gracefully resolve variations to the correct database entity before fetching the topic profile.

2. Vector Space Taxonomy Maps (The FIASCO* Resurrexit)

• The Upgrade: Evaluate embedding the static OpenAlex Topic hierarchy/keywords directly in a low-dimensional vector space (e.g., using a compact transformer model or pre-computed embeddings).
• Why it matters: This would bring back the spatial intuition of the original FIASCO app. Instead of just showing a tabular similarity score, we could plot journals as structural contour maps or 2D density distributions over the global landscape of science, matching the visual elegance of the original prototype without the monumental updating friction.

*FIASCO is my old app from 2023 which worked with a precalculated matrix of overlaps of areas on a 2D histogram like density chart derived from SPECTER embedded content of the journals dimred by UMAP
