#this is FIASCOlight.py just renamed
# FIASCOlight was derived from smartapp.py 


    # Put a safe ceiling on total candidate lookups to protect performance
    #if len(top_candidates) > 250:
     #   top_candidates = top_candidates.head(250) 

#Fine tuning:
#      top_candidates = candidates.head(100).copy()  # Increase pool to process true profiles
#JOURNALS_PER_TOPIC = 100  # Increased from 50 to sweep in more core titles
#MIN_WORKS_IN_TOPIC = 3    # Lowered from 5 to catch journals with slight keyword splits
 # Filter out target journal and keep top XX candidates for full profile evaluation
 #   candidates = candidates[candidates["source_id"] != target_id]
  #  top_candidates = candidates.head(200).copy()

# making it actually usable:

#1. Dynamic Topic Deepness (TOP_N_TOPICS)

#Instead of always pulling exactly 15 topics, check how concentrated the journal actually is.
# The Logic: Keep adding topics until you have captured at least 70% or 80% of the journal's total publishing output.

#2. Dynamic Work Thresholds (MIN_WORKS_IN_TOPIC)

#A static threshold like MIN_WORKS_IN_TOPIC = 5 breaks down if you are analyzing a very small journal or a brand-new publication that has only published 10 papers total since 2023.
#The Logic: Base the inclusion threshold on a percentage of the target journal's activity in that field, rather than a flat number.

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# TUNABLE, ACCURATE & SECURE JOURNAL COMPETITOR FINDER
# ---------------------------------------------------------

st.set_page_config(page_title="CATALEX - Journal Competitor Finder", layout="wide")

BASE = "https://api.openalex.org"

# --- SIDEBAR WIDGETS FOR DYNAMIC FILTERING ---
st.sidebar.title("🎛️ Control Parameters")

TARGET_JOURNAL = st.sidebar.text_input("Target Journal Name", value="PROTEOMICS")
FROM_DATE = st.sidebar.text_input("Filter Works From Date (YYYY-MM-DD)", value="2023-01-01")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Extraction & Threshold Settings")

JOURNALS_PER_TOPIC = st.sidebar.slider(
    label="Max Journals Fetched per Topic",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="Controls the initial broad pool depth retrieved from OpenAlex for each core topic area."
)

min_papers = st.sidebar.slider(
    label="Minimum Papers per Shared Topic",
    min_value=1,
    max_value=50,
    value=5,
    help="Excludes competitor journals that have fewer than this many papers in a target topic field."
)

min_overlap_pct = st.sidebar.slider(
    label="Minimum Similarity Cutoff Score",
    min_value=0.0,
    max_value=50.0,
    value=0.5,
    step=0.5,
    help="Filters out journals from the final display matrix if their similarity score falls below this value."
)

max_candidates = st.sidebar.slider(
    label="Max Candidate Profile Scans Ceiling",
    min_value=50,
    max_value=500,
    value=250,
    step=50,
    help="Puts a safety limit on the total volume of deep secondary profile API lookups performed."
)

whitelist_journals = st.sidebar.multiselect(
    label="📌 Whitelist Journals (Always Include)",
    options=[
        "Journal of Proteomics", 
        "Molecular & Cellular Proteomics", 
        "Proteomics", 
        "Journal of Proteome Research"
    ],
    default=["Journal of Proteomics", "Molecular & Cellular Proteomics"],
    help="Selected journals bypass candidate truncation rules and are guaranteed an evaluation slot."
)

# --- SIDEBAR FOR CREDENTIAL MANAGEMENT ---
st.sidebar.markdown("---")
st.sidebar.title("🔐 API Credentials")
user_api_key = st.sidebar.text_input("OpenAlex API Key", type="password", help="Enter your premium OpenAlex API key.")
user_mailto = st.sidebar.text_input("Polite Pool Email (mailto)", help="Optional: Enter your email address.")

st.title("🐈CATALEX")
st.caption(
    "Cosine Alignment of Topics via OpenAlex - dynamic scope-based only journal competitor finder using OpenAlex true global profiles. "
    f"Target journal: **{TARGET_JOURNAL}**"
)

# ---------------------------------------------------------
# OpenAlex Helpers (Passing keys explicitly to respect caching)
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def oa_get(endpoint, params=None, api_key="", mailto=""):
    url = f"{BASE}/{endpoint}"
    if params is None:
        params = {}
    if api_key.strip():
        params["api_key"] = api_key.strip()
    if mailto.strip():
        params["mailto"] = mailto.strip()
        
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def find_target_source(target_name, api_key="", mailto=""):
    data = oa_get("sources", {"search": target_name, "per-page": 10}, api_key=api_key, mailto=mailto)
    results = data.get("results", [])
    for x in results:
        if x.get("display_name", "").strip().lower() == target_name.lower():
            return x
    if results:
        return results[0]
    raise ValueError(f"Could not find {target_name} in OpenAlex.")


@st.cache_data(show_spinner=False)
def journal_topic_profile(source_id, from_date, api_key="", mailto=""):
    data = oa_get(
        "works",
        {
            "filter": f"primary_location.source.id:{source_id},from_publication_date:{from_date}",
            "group_by": "primary_topic.id",
        },
        api_key=api_key,
        mailto=mailto
    )
    groups = data.get("group_by", [])
    if not groups:
        return pd.DataFrame(columns=["topic_id", "topic", "count", "share"])

    df = pd.DataFrame([
        {
            "topic_id": x["key"].split("/")[-1],
            "topic": x["key_display_name"],
            "count": x["count"],
        }
        for x in groups
    ])
    total = df["count"].sum()
    df["share"] = df["count"] / total if total else 0
    return df.sort_values("count", ascending=False)

@st.cache_data(show_spinner=False)
def journals_for_topic(topic_id, from_date, min_works, max_journals, api_key="", mailto=""):
    data = oa_get(
        "works",
        {
            "filter": f"primary_topic.id:{topic_id},from_publication_date:{from_date},primary_location.source.type:journal",
            "group_by": "primary_location.source.id",
        },
        api_key=api_key,
        mailto=mailto
    )
    rows = []
    for x in data.get("group_by", []):
        if x["count"] >= min_works:
            rows.append({
                "source_id": x["key"].split("/")[-1],
                "journal": x["key_display_name"],
                "topic_count": x["count"],
            })
    return pd.DataFrame(rows).head(max_journals)

def build_candidate_set(target_topics, from_date, baseline_min_papers, max_journals, api_key="", mailto=""):
    candidate_frames = []
    progress = st.progress(0, text="Finding journals sharing target topics...")

    for i, row in target_topics.iterrows():
        # Dynamically scale topic baseline floor relative to widget selection
        dynamic_min_works = max(baseline_min_papers, int(row["count"] * 0.10)) 
        
        temp = journals_for_topic(
            row["topic_id"], 
            from_date=from_date, 
            min_works=dynamic_min_works, 
            max_journals=max_journals,
            api_key=api_key, 
            mailto=mailto
        )
        if not temp.empty:
            temp["topic_id"] = row["topic_id"]
            candidate_frames.append(temp)
        progress.progress((i + 1) / len(target_topics), text=f"Scanning topic {i + 1}/{len(target_topics)}")
    
    progress.empty()
    if not candidate_frames:
        return pd.DataFrame()

    df_all = pd.concat(candidate_frames, ignore_index=True)
    summary = df_all.groupby(["source_id", "journal"]).agg(
        shared_target_topics=("topic_id", "nunique"),
        works_in_target_topics=("topic_count", "sum")
    ).reset_index()
    
    return summary.sort_values(["shared_target_topics", "works_in_target_topics"], ascending=False)

def cosine_similarity(profile_a, profile_b):
    topics = set(profile_a.keys()) | set(profile_b.keys())
    a = np.array([profile_a.get(t, 0.0) for t in topics])
    b = np.array([profile_b.get(t, 0.0) for t in topics])
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------
# MAIN PIPELINE RUNTIME
# ---------------------------------------------------------
try:
    if not TARGET_JOURNAL.strip():
        st.warning("Please type a valid target journal name in the sidebar.")
        st.stop()

    # Resolve target journal with optional credentials
    target = find_target_source(TARGET_JOURNAL, api_key=user_api_key, mailto=user_mailto)
    target_id = target["id"].split("/")[-1]
    st.success(f"Target resolved as: {target['display_name']} ({target_id})")

    with st.spinner(f"Getting true topic profile for {target['display_name']}..."):
        full_target_profile = journal_topic_profile(target_id, from_date=FROM_DATE, api_key=user_api_key, mailto=user_mailto)

    if full_target_profile.empty:
        st.error("No topic data found for this journal in the chosen date range.")
        st.stop()

    # Accumulate slices to map exactly 75% of global output
    full_target_profile["cumulative_share"] = full_target_profile["share"].cumsum()
    threshold_df = full_target_profile[full_target_profile["cumulative_share"] <= 0.75]
    dynamic_n = max(5, min(len(threshold_df) + 1, 20)) 

    target_topics = full_target_profile.head(dynamic_n).reset_index(drop=True)

    # UI Element: Target Profile Layout
    st.subheader(f"{target['display_name']} Structural Field Distribution")
    display_topics = target_topics.copy()
    display_topics["share"] = (display_topics["share"] * 100).round(1)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(
            display_topics[["topic", "count", "share"]].rename(columns={"topic": "Core Topic", "count": "Papers", "share": "%"}),
            use_container_width=True, hide_index=True
        )
    with col2:
        st.bar_chart(display_topics.set_index("topic")["share"])

    # Core Candidate Processing
    st.subheader("Extracting Market Competitors")
    candidates = build_candidate_set(
        target_topics, 
        from_date=FROM_DATE, 
        baseline_min_papers=min_papers, 
        max_journals=JOURNALS_PER_TOPIC, 
        api_key=user_api_key, 
        mailto=user_mailto
    )

    if candidates.empty:
        st.warning("No candidate competitor journals found matching structural parameters.")
        st.stop()

    # Filter out target journal itself
    candidates = candidates[candidates["source_id"] != target_id]
    
    # --- PROVENANCE ENFORCEMENT: APPLY DYNAMIC WHITELIST OVERRIDES ---
    is_whitelisted = candidates["journal"].isin(whitelist_journals)
    whitelisted_pool = candidates[is_whitelisted].copy()
    standard_pool = candidates[~is_whitelisted].copy()

    # Keep standard pool items that overlap with at least 25% of target topics
    min_shared_topics = max(2, int(len(target_topics) * 0.25))
    filtered_standard = standard_pool[standard_pool["shared_target_topics"] >= min_shared_topics].copy()

    # Recombine datasets, giving guaranteed prioritization pass to whitelisted assets
    top_candidates = pd.concat([whitelisted_pool, filtered_standard], ignore_index=True).drop_duplicates(subset=["source_id"])
    top_candidates = top_candidates.sort_values(["shared_target_topics", "works_in_target_topics"], ascending=False)

    if len(top_candidates) > max_candidates:
        top_candidates = top_candidates.head(max_candidates) 

    # Prepare vector arrays
    target_vector_map = dict(zip(full_target_profile["topic_id"], full_target_profile["share"]))
    similarities = []
    matrix_visualization_data = []

    # Inject target line coordinates for heatmap mapping
    for _, t_row in target_topics.iterrows():
        matrix_visualization_data.append({
            "Source_ID": target_id,
            "Topic": t_row["topic"],
            "Share (%)": round(target_vector_map.get(t_row["topic_id"], 0.0) * 100, 1)
        })

    progress = st.progress(0, text="Analyzing true candidate profiles and running cosine vectors...")
    total_cand = len(top_candidates)

    for n, (_, row) in enumerate(top_candidates.iterrows()):
        cand_profile_df = journal_topic_profile(row["source_id"], from_date=FROM_DATE, api_key=user_api_key, mailto=user_mailto)
        cand_vector_map = dict(zip(cand_profile_df["topic_id"], cand_profile_df["share"]))
        
        sim = cosine_similarity(target_vector_map, cand_vector_map)
        similarities.append(sim)

        for _, t_row in target_topics.iterrows():
            matrix_visualization_data.append({
                "Source_ID": row["source_id"],
                "Topic": t_row["topic"],
                "Share (%)": round(cand_vector_map.get(t_row["topic_id"], 0.0) * 100, 1)
            })

        progress.progress((n + 1) / total_cand, text=f"Processing profile {n + 1}/{total_cand}")
    
    progress.empty()

    top_candidates["Similarity"] = (np.array(similarities) * 100).round(1)
    
    # --- FILTER BY MINIMUM SIMILARITY OVERLAP PERCENTAGE (EXCEPT WHITELIST) ---
    passed_filter = (top_candidates["Similarity"] >= min_overlap_pct) | (top_candidates["journal"].isin(whitelist_journals))
    top_candidates = top_candidates[passed_filter]
    top_candidates = top_candidates.sort_values("Similarity", ascending=False).head(30)

    # ---------------------------------------------------------
    # Visualizations Section
    # ---------------------------------------------------------
    st.subheader("Visual Scope Diagnostics")
    viz_tab1, viz_tab2 = st.tabs(["📊 Scope Overlay Heatmap", "🕸️ Scope Overlap Radar Map"])

    id_to_name = dict(zip(top_candidates["source_id"], top_candidates["journal"]))
    id_to_name[target_id] = f"TARGET: {target['display_name']}"

    df_heatmap = pd.DataFrame(matrix_visualization_data)
    valid_journals = list(top_candidates["source_id"].unique()) + [target_id]
    df_heatmap = df_heatmap[df_heatmap["Source_ID"].isin(valid_journals)]

    with viz_tab1:
        st.markdown("**True Scope Distribution Matrix:** Explicitly tracks concentration variables relative to core focus domains.")
        pivot_hm = df_heatmap.pivot(index="Source_ID", columns="Topic", values="Share (%)").fillna(0)
        
        heatmap_order = [target_id] + list(top_candidates["source_id"].unique())
        pivot_hm = pivot_hm.reindex(heatmap_order)
        pivot_hm.index = [id_to_name.get(x, x) for x in pivot_hm.index]
        
        fig_hm = px.imshow(
            pivot_hm, text_auto=True, aspect="auto",
            labels=dict(x="Topic Fields", y="Competitor Journals", color="Global Share %"),
            color_continuous_scale="Viridis"
        )
        fig_hm.update_yaxes(type='category', dtick=1)
        fig_hm.update_layout(height=800, margin=dict(l=200, r=20, t=20, b=150))
        st.plotly_chart(fig_hm, use_container_width=True)

    with viz_tab2:
        st.markdown("**Fingerprint alignment shapes:** Select competitors to trace architectural alignment directly against baseline shapes.")
        journal_options = list(top_candidates["source_id"].unique())
        selected_ids = st.multiselect(
            "Select journals to trace:", 
            options=journal_options, 
            default=journal_options[:2] if len(journal_options) >= 2 else journal_options,
            format_func=lambda x: id_to_name.get(x, x)
        )
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=target_topics["share"] * 100, theta=target_topics["topic"],
            fill='toself', fillcolor='rgba(26, 26, 26, 0.15)',  name=f"TARGET: {target['display_name']}", 
            line=dict(
                color='#1A1A1A',                 # Deep Charcoal/Black to pop against colorful competitors
                width=4,                         # Extra thick line
                dash='solid'                     # Solid baseline anchor
            ),
            marker=dict(
                size=8, 
                color='#1A1A1A', 
                symbol='circle'                  # Adds visible physical vertices
            )
        ))
        
        for s_id in selected_ids:
            sub_sj = df_heatmap[df_heatmap["Source_ID"] == s_id].copy()
            r_values = []
            for topic_name in target_topics["topic"]:
                match = sub_sj[sub_sj["Topic"] == topic_name]
                val = float(match["Share (%)"].values[0]) if not match.empty else 0.0
                r_values.append(val)
                
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values, theta=target_topics["topic"],
                fill='toself', name=id_to_name.get(s_id, s_id)
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(target_topics["share"]*100) + 5])),
            showlegend=True, height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Output Data Display Elements
    st.subheader("Competitor Ranking Summary Matrix")
    st.dataframe(
        top_candidates[["journal", "Similarity", "shared_target_topics", "works_in_target_topics"]].rename(
            columns={
                "journal": "Journal Title Space",
                "shared_target_topics": "Overlapping Core Topics",
                "works_in_target_topics": "Total Intersecting Publications"
            }
        ), use_container_width=True, hide_index=True
    )

except Exception as e:
    st.error(f"Pipeline Runtime Exception: {str(e)}")
