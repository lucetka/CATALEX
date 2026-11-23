#CATALEXv2.0.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# TUNABLE, ACCURATE & SECURE JOURNAL COMPETITOR FINDER
# ---------------------------------------------------------

st.set_page_config(
    page_title="CATALEX - Journal Competitor Finder",
    layout="wide"
)

BASE = "https://api.openalex.org"

# ---------------------------------------------------------
# Helper for self-contained Plotly HTML downloads
# ---------------------------------------------------------

def plotly_html_bytes(fig):
    """
    Return a completely self-contained Plotly HTML file.
    Plotly.js is embedded directly in the HTML.
    """
    return fig.to_html(
        full_html=True,
        include_plotlyjs=True
    ).encode("utf-8")


# ---------------------------------------------------------
# SIDEBAR WIDGETS FOR DYNAMIC FILTERING
# ---------------------------------------------------------

st.sidebar.title("🎛️ Control Parameters")

TARGET_JOURNAL = st.sidebar.text_input(
    "Target Journal Name",
    value="PROTEOMICS"
)

FROM_DATE = st.sidebar.text_input(
    "Filter Works From Date (YYYY-MM-DD)",
    value="2023-01-01"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Vector & Similarity Settings")

SIM_METRIC = st.sidebar.selectbox(
    "Similarity Metric",
    options=["Cosine Similarity", "TF-IDF Cosine"],
    index=0,
    help="Cosine Similarity relies strictly on raw shares. TF-IDF down-weights hyper-broad generic topics across OpenAlex."
)

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
    label="Minimum Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Filters competitors by alignment scores. 0 = no alignment; 1 = identical profile."
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
    default=[
        "Journal of Proteomics",
        "Molecular & Cellular Proteomics"
    ],
    help="Selected journals bypass candidate truncation rules and are guaranteed an evaluation slot."
)

max_shown = st.sidebar.slider(
    label="Max Competitors Shown in Dashboard",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help="Limits the maximum number of high-ranked competitors populated inside your tables, heatmaps, and dropdown menus."
)


# ---------------------------------------------------------
# SIDEBAR FOR CREDENTIAL MANAGEMENT
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.title("🔐 API Credentials")

user_api_key = st.sidebar.text_input(
    "OpenAlex API Key",
    type="password",
    help="Enter your premium OpenAlex API key."
)

user_mailto = st.sidebar.text_input(
    "Polite Pool Email (mailto)",
    help="Optional: Enter your email address."
)

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("🐈 CATALEX")
st.caption(
    "Cosine Alignment of Topics via OpenAlex - dynamic scope-based only "
    f"journal competitor finder using OpenAlex global profiles. Target journal: **{TARGET_JOURNAL}**"
)

# ---------------------------------------------------------
# OpenAlex Helpers
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


######### END chunk 1
@st.cache_data(show_spinner=False)
def journal_topic_profile(source_id, from_date, api_key="", mailto=""):
    date_filter = f"from_publication_date:{from_date}" if "-" in from_date and ":" not in from_date else from_date
    data = oa_get(
        "works",
        {
            "filter": f"primary_location.source.id:{source_id},{date_filter}",
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


# ---------------------------------------------------------
# MATHEMATICAL ALIGNMENT IMPLEMENTATIONS
# ---------------------------------------------------------

def calculate_similarity(profile_a, profile_b, metric="Cosine Similarity"):
    topics = set(profile_a.keys()) | set(profile_b.keys())
    a = np.array([profile_a.get(t, 0.0) for t in topics])
    b = np.array([profile_b.get(t, 0.0) for t in topics])
    
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compute_tfidf_vectors(target_profile, candidate_profiles):
    all_documents = [target_profile] + list(candidate_profiles.values())
    total_docs = len(all_documents)
    
    doc_freq = {}
    for doc in all_documents:
        for topic in doc.keys():
            if doc[topic] > 0:
                doc_freq[topic] = doc_freq.get(topic, 0) + 1
                
    idf = {topic: np.log(total_docs / (1 + freq)) + 1.0 for topic, freq in doc_freq.items()}
    
    weighted_target = {t: share * idf.get(t, 1.0) for t, share in target_profile.items()}
    weighted_candidates = {}
    for s_id, profile in candidate_profiles.items():
        weighted_candidates[s_id] = {t: share * idf.get(t, 1.0) for t, share in profile.items()}
        
    return weighted_target, weighted_candidates


# ---------------------------------------------------------
# MAIN PIPELINE RUNTIME
# ---------------------------------------------------------

try:
    if not TARGET_JOURNAL.strip():
        st.warning("Please type a valid target journal name in the sidebar.")
        st.stop()

    target = find_target_source(TARGET_JOURNAL, api_key=user_api_key, mailto=user_mailto)
    target_id = target["id"].split("/")[-1]

    st.success(f"Target resolved as: {target['display_name']} ({target_id})")

    # Initialize evolution mode flags in session state
    if "use_evolution_vector" not in st.session_state:
        st.session_state.use_evolution_vector = False
    if "evolution_target_vector" not in st.session_state:
        st.session_state.evolution_target_vector = None
    if "evolution_epoch_label" not in st.session_state:
        st.session_state.evolution_epoch_label = ""

    # -----------------------------------------------------
    # UPGRADED SUB-ANALYSIS: ADVANCED TEMPORAL EVOLUTION MODULE
    # -----------------------------------------------------
    st.markdown("---")
    with st.expander("⏳ Analyze Journal Scope Evolution in Time", expanded=False):
        st.subheader("Temporal Profile Vector Evolution")
        st.caption("Tracks how a journal's topic distribution and competitive landscape shift over time.")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            epoch_1 = st.text_input("Epoch 1 Window (Baseline Years)", value="2016-2020")
        with col_t2:
            epoch_2 = st.text_input("Epoch 2 Window (Recent Years)", value="2021-2026")
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            calc_pressed = st.button("Calculate Core Temporal Trajectory")
        with col_btn2:
            st.checkbox(
                "🚀 Activate 'Chasing Mode': Rank competitors strictly using fast-growing Epoch 2 vector",
                key="use_evolution_vector",
                help="When enabled, the alignment engine switches targets from the baseline data to your high-growth frontiers."
            )
            
        if calc_pressed or st.session_state.use_evolution_vector:
            with st.spinner("Fetching temporal maps via OpenAlex..."):
                prof_e1 = journal_topic_profile(target_id, from_date=f"publication_year:{epoch_1}", api_key=user_api_key, mailto=user_mailto)
                prof_e2 = journal_topic_profile(target_id, from_date=f"publication_year:{epoch_2}", api_key=user_api_key, mailto=user_mailto)
                
            if prof_e1.empty or prof_e2.empty:
                st.error("Insufficient publication volume found within one or both configured periods.")
            else:
                v1_map = dict(zip(prof_e1["topic_id"], prof_e1["share"]))
                v2_map = dict(zip(prof_e2["topic_id"], prof_e2["share"]))
                
                t_similarity = calculate_similarity(v1_map, v2_map, metric="Cosine Similarity")
                drift_score = round(1.0 - t_similarity, 4)
                
                kpi1, kpi2 = st.columns(2)
                with kpi1:
                    st.metric("Internal Scope Drift Score", drift_score)
                with kpi2:
                    stability_status = "Stable Editorial Scope" if drift_score < 0.15 else "Evolving Strategic Footprint"
                    st.info(f"Detected Vector Trend: **{stability_status}**")
                
                all_t_topics = set(prof_e1["topic_id"]) | set(prof_e2["topic_id"])
                name_map = {**dict(zip(prof_e1["topic_id"], prof_e1["topic"])), **dict(zip(prof_e2["topic_id"], prof_e2["topic"]))}
                
                delta_rows = []
                # Isolate positive growth vectors to construct the dynamic future target shape
                growth_vector = {}
                for tid in all_t_topics:
                    s1 = v1_map.get(tid, 0.0)
                    s2 = v2_map.get(tid, 0.0)
                    diff = s2 - s1
                    delta_rows.append({"Topic": name_map.get(tid, tid), "Share Delta (%)": round(diff * 100, 2)})
                    if diff > 0:
                        growth_vector[tid] = diff

                # Normalize growth vector so its components sum to 1.0
                growth_sum = sum(growth_vector.values())
                if growth_sum > 0:
                    st.session_state.dynamic_growth_vector = {k: v / growth_sum for k, v in growth_vector.items()}
                else:
                    st.session_state.dynamic_growth_vector = v2_map
                
                st.session_state.evolution_epoch_label = f"Growth Vector ({epoch_1} ➡️ {epoch_2})"
                
                df_delta = pd.DataFrame(delta_rows).sort_values("Share Delta (%)", ascending=False)
                df_viz = pd.concat([df_delta.head(7), df_delta.tail(7)]).drop_duplicates().sort_values("Share Delta (%)")
                
                fig_waterfall = px.bar(
                    df_viz, x="Share Delta (%)", y="Topic",
                    orientation="h",
                    title=f"Editorial Divergence: Top Fields Shifts ({epoch_1} ➡️ {epoch_2})",
                    color="Share Delta (%)",
                    color_continuous_scale="RdBu",
                    text_auto=True
                )
                fig_waterfall.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
                st.plotly_chart(fig_waterfall, use_container_width=True)

#### end chunk 2

    with st.spinner(f"Getting true topic profile for {target['display_name']}..."):
        full_target_profile = journal_topic_profile(target_id, from_date=FROM_DATE, api_key=user_api_key, mailto=user_mailto)

    if full_target_profile.empty:
        st.error("No topic data found for this journal in the chosen date range.")
        st.stop()

    full_target_profile["cumulative_share"] = full_target_profile["share"].cumsum()
    threshold_df = full_target_profile[full_target_profile["cumulative_share"] <= 0.75]
    dynamic_n = max(5, min(len(threshold_df) + 1, 20))
    target_topics = full_target_profile.head(dynamic_n).reset_index(drop=True)

    # Establish descriptive timeframe annotations based on configuration states
    if st.session_state.use_evolution_vector and "dynamic_growth_vector" in st.session_state:
        target_vector_map = st.session_state.dynamic_growth_vector
        active_timeframe_string = f"Strict Target Evolution: {st.session_state.evolution_epoch_label}"
        st.info("💡 **Active Profile Mode:** Tracking competitors currently aligning with your highest expanding growth vectors.")
    else:
        target_vector_map = dict(zip(full_target_profile["topic_id"], full_target_profile["share"]))
        active_timeframe_string = f"Standard Matrix (Works from: {FROM_DATE})"

    st.subheader(f"{target['display_name']} Structural Field Distribution")
    st.caption(f"Currently viewing: **{active_timeframe_string}**")
    
    display_topics = target_topics.copy()
    display_topics["share"] = (display_topics["share"] * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            display_topics[["topic", "count", "share"]].rename(
                columns={"topic": "Core Topic", "count": "Papers", "share": "%"}
            ),
            use_container_width=True, hide_index=True
        )
    with col2:
        fig_bar = px.bar(
            display_topics, x="share", y="topic", orientation="h",
            labels={"share": "Share (%)", "topic": "Core Topic"},
            title=f"Core Topic Architecture — {active_timeframe_string}"
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Extracting Market Competitors")
    candidates = build_candidate_set(
        target_topics, from_date=FROM_DATE, baseline_min_papers=min_papers,
        max_journals=JOURNALS_PER_TOPIC, api_key=user_api_key, mailto=user_mailto
    )

    if candidates.empty:
        st.warning("No candidate competitor journals found matching structural parameters.")
        st.stop()

    candidates = candidates[candidates["source_id"] != target_id]

    is_whitelisted = candidates["journal"].isin(whitelist_journals)
    whitelisted_pool = candidates[is_whitelisted].copy()
    standard_pool = candidates[~is_whitelisted].copy()

    min_shared_topics = max(2, int(len(target_topics) * 0.25))
    filtered_standard = standard_pool[standard_pool["shared_target_topics"] >= min_shared_topics].copy()

    top_candidates = pd.concat([whitelisted_pool, filtered_standard], ignore_index=True).drop_duplicates(subset=["source_id"])
    top_candidates = top_candidates.sort_values(["shared_target_topics", "works_in_target_topics"], ascending=False)

    if len(top_candidates) > max_candidates:
        top_candidates = top_candidates.head(max_candidates)

    candidate_profiles = {}
    st.markdown("##### 🔄 Processing Competitor Vector Profiles")
    
    def fetch_profile_worker(source_id, display_name):
        try:
            profile_df = journal_topic_profile(source_id, from_date=FROM_DATE, api_key=user_api_key, mailto=user_mailto)
            if profile_df.empty:
                return source_id, {}
            return source_id, dict(zip(profile_df["topic_id"], profile_df["share"]))
        except Exception:
            return source_id, {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(fetch_profile_worker, row["source_id"], row["journal"]): row 
            for _, row in top_candidates.iterrows()
        }
        
        c_progress = st.progress(0, text="Initializing concurrent lookups...")
        completed = 0
        total_futures = len(futures)
        
        for future in as_completed(futures):
            s_id, profile_map = future.result()
            completed += 1
            if profile_map:
                candidate_profiles[s_id] = profile_map
            c_progress.progress(completed / total_futures, text=f"Fetched profile tracking: {completed}/{total_futures}")
        c_progress.empty()

    similarity_scores = {}
    if SIM_METRIC == "TF-IDF Cosine":
        weighted_target, weighted_candidates = compute_tfidf_vectors(target_vector_map, candidate_profiles)
        for s_id in candidate_profiles.keys():
            similarity_scores[s_id] = calculate_similarity(weighted_target, weighted_candidates[s_id])
    else:
        for s_id, profile_map in candidate_profiles.items():
            similarity_scores[s_id] = calculate_similarity(target_vector_map, profile_map)

    top_candidates["Similarity"] = top_candidates["source_id"].map(similarity_scores).fillna(0.0).round(3)

    passed_filter = (top_candidates["Similarity"] >= min_overlap_pct) | (top_candidates["journal"].isin(whitelist_journals))
    top_candidates = top_candidates[passed_filter].sort_values("Similarity", ascending=False).head(max_shown)

    matrix_visualization_data = []
    for _, t_row in target_topics.iterrows():
        matrix_visualization_data.append({
            "Source_ID": target_id, "Topic": t_row["topic"], 
            "Share (%)": round(target_vector_map.get(t_row["topic_id"], 0.0) * 100, 1)
        })

    for _, row in top_candidates.iterrows():
        cand_map = candidate_profiles.get(row["source_id"], {})
        for _, t_row in target_topics.iterrows():
            matrix_visualization_data.append({
                "Source_ID": row["source_id"], "Topic": t_row["topic"],
                "Share (%)": round(cand_map.get(t_row["topic_id"], 0.0) * 100, 1)
            })

    # UPGRADED: Explicit visual anchoring context for diagnostics header
    st.subheader(f"📊 Visual Scope Diagnostics — {active_timeframe_string}")
    viz_tab1, viz_tab2 = st.tabs(["📊 Scope Overlay Heatmap Matrix", "🕸️ Scope Overlap Radar Map"])

    id_to_name = dict(zip(top_candidates["source_id"], top_candidates["journal"]))
    id_to_name[target_id] = f"TARGET: {target['display_name']} ({active_timeframe_string})"

    df_heatmap = pd.DataFrame(matrix_visualization_data)
    valid_journals = list(top_candidates["source_id"].unique()) + [target_id]
    df_heatmap = df_heatmap[df_heatmap["Source_ID"].isin(valid_journals)]

    with viz_tab1:
        st.markdown(f"**True Scope Distribution Matrix ({active_timeframe_string}):** Values mapped as percentages of entire portfolio concentration.")
        pivot_hm = df_heatmap.pivot(index="Source_ID", columns="Topic", values="Share (%)").fillna(0)
        heatmap_order = [target_id] + list(top_candidates["source_id"].unique())
        pivot_hm = pivot_hm.reindex(heatmap_order)
        pivot_hm.index = [id_to_name.get(x, x) for x in pivot_hm.index]

        fig_hm = px.imshow(
            pivot_hm, text_auto=True, aspect="auto",
            labels=dict(x="Topic Fields", y="Competitor Journals", color="Global Share %"),
            color_continuous_scale="Viridis",
            title=f"Scope Overlap Architecture ({active_timeframe_string})"
        )
        fig_hm.update_yaxes(type="category", dtick=1)
        fig_hm.update_layout(height=800, margin=dict(l=200, r=20, t=50, b=150))
        st.plotly_chart(fig_hm, use_container_width=True)

    with viz_tab2:
        st.markdown(f"**Fingerprint alignment shapes ({active_timeframe_string}):** Trace specific architectural alignments against baseline models.")
        journal_options = list(top_candidates["source_id"].unique())
        selected_ids = st.multiselect(
            "Select journals to trace:", options=journal_options,
            default=journal_options[:2] if len(journal_options) >= 2 else journal_options,
            format_func=lambda x: id_to_name.get(x, x)
        )

        fig_radar = go.Figure()
        
        # Pull actual baseline display metric arrays based on active scope configuration
        radar_target_shares = [target_vector_map.get(tid, 0.0) * 100 for tid in target_topics["topic_id"]]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_target_shares, theta=target_topics["topic"], fill="toself",
            fillcolor="rgba(26, 26, 26, 0.15)", name=f"TARGET Baseline ({active_timeframe_string})",
            line=dict(color="#1A1A1A", width=4), marker=dict(size=8)
        ))

        for s_id in selected_ids:
            sub_sj = df_heatmap[df_heatmap["Source_ID"] == s_id].copy()
            r_values = []
            #for topic_name in target_topics["topic"]:
            #    match = sub_sj[sub_sj["Topic"] == topic_name]
            #    val = float(match["Share (%)"].values) if not match.empty else 0.0
            #    r_values.append(val)

            # as the above was working on Windows but failing on streamlit cloud (unix?) here a more robust way:
            for topic_name in target_topics["topic"]:
                match = sub_sj[sub_sj["Topic"] == topic_name]
                val = float(match["Share (%)"].item()) if not match.empty else 0.0
                r_values.append(val)


            fig_radar.add_trace(go.Scatterpolar(
                r=r_values, theta=target_topics["topic"], fill="toself", name=id_to_name.get(s_id, s_id)
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(radar_target_shares) + 5])),
            title=f"Fingerprint Comparison Map — {active_timeframe_string}",
            showlegend=True, height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader(f"Competitor Ranking Summary Matrix — {active_timeframe_string}")
    st.dataframe(
        top_candidates[["journal", "Similarity", "shared_target_topics", "works_in_target_topics"]].rename(
            columns={"journal": "Journal Title Space", "Similarity": f"{SIM_METRIC}",
                        "shared_target_topics": "Overlapping Core Topics", "works_in_target_topics": "Total Intersecting Publications"
                }
        ),
        use_container_width=True, hide_index=True
                )
    
    st.subheader("🔎 Competitor Query String Generator")
    if not top_candidates.empty:
        max_query_journals = len(top_candidates)
        top_x = st.number_input("Number of top competitors to include", min_value=1, max_value=max_query_journals, value=min(3, max_query_journals), step=1)
        query_journals = top_candidates.head(int(top_x))["journal"].tolist()
        boolean_query = "(" + " OR ".join(f'"{journal}"' for journal in query_journals) + ")"
        st.text_area("Generated query", value=boolean_query, height=100)

except Exception as e:
    st.error(f"Pipeline Runtime Exception: {str(e)}")
