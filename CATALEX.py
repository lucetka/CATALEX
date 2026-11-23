# CATALEX v2.1 - optional user-supplied journal metric filtering

import streamlit as st
import requests
import re
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# CATALEX
# ---------------------------------------------------------

st.set_page_config(
    page_title="CATALEX",
    layout="wide"
)

BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT = 60
MAX_API_ATTEMPTS = 4

def publication_filter(value):
    """Build a validated OpenAlex date filter from YYYY-MM-DD or YYYY-YYYY."""
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"from_publication_date:{value}"
    match = re.fullmatch(r"(\d{4})-(\d{4})", value)
    if match:
        start_year, end_year = map(int, match.groups())
        if start_year > end_year:
            raise ValueError("The start year must not be later than the end year.")
        return (f"from_publication_date:{start_year}-01-01,"
                f"to_publication_date:{end_year}-12-31")
    raise ValueError("Use YYYY-MM-DD for a start date or YYYY-YYYY for a year window.")

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
# OPTIONAL USER-SUPPLIED JOURNAL METRIC FILTER
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Optional Journal Metric Filter")
metric_file = st.sidebar.file_uploader(
    "Upload journal metrics (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="The file stays in the current Streamlit session. Choose any numeric journal metric, such as JIF, JCI, CiteScore, SJR or a custom score."
)

metric_df_raw = None
metric_sheet = None
metric_title_col = None
metric_value_col = None
metric_issn_col = None
metric_min_text = ""
metric_max_text = ""
include_unmatched_metric = True
metric_duplicate_rule = "Maximum"

if metric_file is not None:
    try:
        if metric_file.name.lower().endswith(".csv"):
            metric_df_raw = pd.read_csv(metric_file)
        else:
            excel_book = pd.ExcelFile(metric_file)
            metric_sheet = st.sidebar.selectbox("Worksheet", excel_book.sheet_names)
            metric_df_raw = pd.read_excel(metric_file, sheet_name=metric_sheet)

        usable_columns = [str(c) for c in metric_df_raw.columns]
        metric_df_raw.columns = usable_columns
        if not usable_columns:
            raise ValueError("The uploaded table has no columns.")

        metric_title_col = st.sidebar.selectbox(
            "Full journal title column", usable_columns,
            help="Used as the main fallback match when ISSN is unavailable or does not match."
        )
        metric_value_col = st.sidebar.selectbox(
            "Numeric metric column", usable_columns,
            index=min(1, len(usable_columns) - 1)
        )
        metric_issn_choice = st.sidebar.selectbox(
            "ISSN column (optional)", ["None"] + usable_columns
        )
        metric_issn_col = None if metric_issn_choice == "None" else metric_issn_choice
        metric_duplicate_rule = st.sidebar.selectbox(
            "If a journal occurs more than once", ["Maximum", "Mean", "First"]
        )
        col_metric_min, col_metric_max = st.sidebar.columns(2)
        with col_metric_min:
            metric_min_text = st.text_input("Minimum", value="", help="Leave blank for no minimum.")
        with col_metric_max:
            metric_max_text = st.text_input("Maximum", value="", help="Leave blank for no maximum.")
        include_unmatched_metric = st.sidebar.checkbox(
            "Keep journals with no metric match", value=True,
            help="Turn this off to restrict results strictly to journals matched in the uploaded table."
        )
    except Exception as metric_upload_error:
        st.sidebar.error(f"Could not read metric file: {metric_upload_error}")
        metric_df_raw = None

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
#st.caption(
#    "Cosine Alignment of Topics via OpenAlex - dynamic usinf real time data from OpenAlex; identifies competitors purely by topical scope, with optional runtime filtering for custom metrics (e.g., Impact Factor) via file upload."
#    f"Target journal: **{TARGET_JOURNAL}**"
#)

#st.subheader("***C***osine ***A***lignment of ***T***opics via Open***Alex***: Scope-Based Journal Competitor Finder")
st.subheader(":blue[C]osine :blue[A]lignment of :blue[T]opics via Open:blue[ALEX]: Scope-Based Journal Competitor Finder")

st.markdown(
    "Identifies competitors using **Cosine Alignment of Topics** on real-time **OpenAlex** data. "
    "Matches purely on topical scope by default, with optional runtime filtering for custom metrics "
    "(e.g., Impact Factor) via file upload."
)
st.caption(f"🎯 Currently selected target journal: **{TARGET_JOURNAL}**")


# ---------------------------------------------------------
# OpenAlex Helpers
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def oa_get(endpoint, params=None, api_key="", mailto=""):
    url = f"{BASE}/{endpoint}"
    # Copy parameters rather than mutating the caller's dictionary.
    request_params = dict(params or {})
    if api_key.strip():
        request_params["api_key"] = api_key.strip()
    if mailto.strip():
        request_params["mailto"] = mailto.strip()

    # Retry only transient rate-limit/server failures with bounded backoff.
    for attempt in range(MAX_API_ATTEMPTS):
        response = requests.get(url, params=request_params, timeout=DEFAULT_TIMEOUT)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response.json()
        if attempt == MAX_API_ATTEMPTS - 1:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After", "")
        time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** attempt)


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
def journal_topic_profile(source_id, date_value, api_key="", mailto=""):
    """Return the primary-topic distribution for one journal and date window."""
    date_filter = publication_filter(date_value)
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
def journals_for_topic(topic_id, date_value, min_works, max_journals, api_key="", mailto=""):
    """Return candidate journals with sufficient volume in one target topic."""
    date_filter = publication_filter(date_value)
    data = oa_get(
        "works",
        {
            "filter": f"primary_topic.id:{topic_id},{date_filter},primary_location.source.type:journal",
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
            date_value=from_date,
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


def normalize_journal_title(value):
    """Normalize titles conservatively for fallback matching."""
    value = str(value or "").casefold().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def normalize_issn(value):
    """Return an eight-character ISSN key without punctuation."""
    value = re.sub(r"[^0-9Xx]", "", str(value or "")).upper()
    return value if len(value) == 8 else ""

def parse_optional_bound(value, label):
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"{label} must be a number or blank.") from exc

def prepare_metric_lookup(df, title_col, value_col, issn_col=None, duplicate_rule="Maximum"):
    """Create title and ISSN lookup tables for any user-selected numeric journal metric."""
    metric = df[[title_col, value_col] + ([issn_col] if issn_col else [])].copy()
    metric["_metric_value"] = pd.to_numeric(metric[value_col], errors="coerce")
    metric["_title_key"] = metric[title_col].map(normalize_journal_title)
    metric["_issn_key"] = metric[issn_col].map(normalize_issn) if issn_col else ""
    metric = metric[metric["_metric_value"].notna() & (metric["_title_key"] != "")].copy()
    if metric.empty:
        raise ValueError("No numeric metric values were found in the selected metric column.")
    aggregation = {"Maximum": "max", "Mean": "mean", "First": "first"}[duplicate_rule]
    title_lookup = metric.groupby("_title_key")["_metric_value"].agg(aggregation).to_dict()
    issn_lookup = (metric[metric["_issn_key"] != ""]
                   .groupby("_issn_key")["_metric_value"].agg(aggregation).to_dict())
    return title_lookup, issn_lookup, len(metric)

@st.cache_data(show_spinner=False)
def source_identity(source_id, api_key="", mailto=""):
    """Fetch the OpenAlex title and ISSNs needed for external metric matching."""
    source = oa_get(f"sources/{source_id}", api_key=api_key, mailto=mailto)
    return {
        "source_id": source_id,
        "openalex_title": source.get("display_name", ""),
        "openalex_issns": source.get("issn") or [],
    }

def attach_user_metric(candidates, title_lookup, issn_lookup, api_key="", mailto=""):
    """Attach a metric by ISSN first and normalized full title second."""
    identities = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(source_identity, sid, api_key, mailto): sid
                   for sid in candidates["source_id"].tolist()}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                identities[sid] = future.result()
            except Exception:
                identities[sid] = {"source_id": sid, "openalex_title": "", "openalex_issns": []}

    values, methods = [], []
    for _, row in candidates.iterrows():
        identity = identities.get(row["source_id"], {})
        matched_value, matched_method = None, "Not matched"
        for issn in identity.get("openalex_issns", []):
            key = normalize_issn(issn)
            if key and key in issn_lookup:
                matched_value, matched_method = issn_lookup[key], "ISSN"
                break
        if matched_value is None:
            title_key = normalize_journal_title(identity.get("openalex_title") or row["journal"])
            if title_key in title_lookup:
                matched_value, matched_method = title_lookup[title_key], "Title"
        values.append(matched_value)
        methods.append(matched_method)
    result = candidates.copy()
    result["User Metric"] = values
    result["Metric Match"] = methods
    return result

# ---------------------------------------------------------
# MATH
# ---------------------------------------------------------

def calculate_similarity(profile_a, profile_b):
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
    #  TEMPORAL EVOLUTION MODULE
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
                "🚀 Activate 'Chasing Mode': Rank competitors using topics that gained share",
                key="use_evolution_vector",
                help="Uses only topics whose portfolio share increased from Epoch 1 to Epoch 2. Each positive change is normalized against all positive changes to form the competitor-matching target vector."
            )
            
        if calc_pressed or st.session_state.use_evolution_vector:
            with st.spinner("Fetching temporal maps via OpenAlex..."):
                prof_e1 = journal_topic_profile(target_id, date_value=epoch_1, api_key=user_api_key, mailto=user_mailto)
                prof_e2 = journal_topic_profile(target_id, date_value=epoch_2, api_key=user_api_key, mailto=user_mailto)
                
            if prof_e1.empty or prof_e2.empty:
                st.error("Insufficient publication volume found within one or both configured periods.")
            else:
                v1_map = dict(zip(prof_e1["topic_id"], prof_e1["share"]))
                v2_map = dict(zip(prof_e2["topic_id"], prof_e2["share"]))
                
                t_similarity = calculate_similarity(v1_map, v2_map)
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
                
                st.session_state.evolution_epoch_label = f"Normalized Positive-Growth Vector ({epoch_1} ➡️ {epoch_2})"
                
                df_delta = pd.DataFrame(delta_rows).sort_values("Share Delta (%)", ascending=False)
                df_viz = pd.concat([df_delta.head(7), df_delta.tail(7)]).drop_duplicates().sort_values("Share Delta (%)")
                
                fig_waterfall = px.bar(
                    df_viz, x="Share Delta (%)", y="Topic",
                    orientation="h",
                    title=f"Change in Topic Share: Largest Shifts ({epoch_1} ➡️ {epoch_2})",
                    color="Share Delta (%)",
                    color_continuous_scale="RdBu",
                    text_auto=True
                )
                fig_waterfall.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
                fig_waterfall.update_xaxes(title="Change in portfolio share (percentage points)")
                fig_waterfall.update_traces(
                    hovertemplate="<b>%{y}</b><br>Change in portfolio share: %{x:.2f} percentage points<extra></extra>"
                )
                st.caption("Bars show the change in each topic's share of the target journal portfolio: Epoch 2 share minus Epoch 1 share. Example: +4.9 means an increase of 4.9 percentage points, not 4.9% relative growth.")
                st.plotly_chart(fig_waterfall, use_container_width=True)
                with st.expander("ℹ️ How to interpret the temporal analysis"):
                    st.markdown(f"""
**What this section asks:** Has the target journal's topic mix changed between **{epoch_1}** and **{epoch_2}**?

- **Internal Scope Drift Score** compares the *complete* topic-share vectors in the two periods. A value near 0 means the overall topic mix is very similar; larger values mean greater structural change. The current *stable/changing* label is a practical diagnostic, not a validated universal threshold.
- **Change in Topic Share** is **Epoch 2 share minus Epoch 1 share**, measured in **percentage points**.
- Example: if a topic rises from 9.8% to 14.7% of the journal portfolio, the chart shows approximately **+4.9 percentage points**. This is not 4.9% relative growth.
- Negative bars mean the topic occupies a smaller share of the journal in Epoch 2. This does not necessarily mean fewer papers were published in absolute terms.

**Use this view for:** identifying fields becoming relatively more or less prominent within the journal.
""")


    with st.spinner(f"Getting true topic profile for {target['display_name']}..."):
        full_target_profile = journal_topic_profile(target_id, date_value=FROM_DATE, api_key=user_api_key, mailto=user_mailto)

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
        st.info("💡 **Active Profile Mode:** Tracking established competitors whose Epoch 2 portfolios align with your target journal's positive-growth vector.")
    else:
        target_vector_map = dict(zip(full_target_profile["topic_id"], full_target_profile["share"]))
        active_timeframe_string = f"Standard Matrix (Works from: {FROM_DATE})"

    if st.session_state.use_evolution_vector and "dynamic_growth_vector" in st.session_state:
        st.subheader(f"{target['display_name']} — Composition of Positive Topic-Share Growth")
        st.caption("The percentages below are normalized growth-vector weights: each positive topic-share change divided by the sum of all positive changes. They sum to 100% and are used for competitor matching; they are NOT the topic shares in Epoch 2.")
    else:
        st.subheader(f"{target['display_name']} Structural Field Distribution")
        st.caption("The percentages below show each topic's share of the target journal portfolio for the selected standard time window.")
    st.caption(f"Currently viewing: **{active_timeframe_string}**")
    
    display_topics = target_topics.copy()
    display_topics["share"] = (display_topics["share"] * 100).round(1)
    share_label = "Share of total positive growth (%)" if st.session_state.use_evolution_vector else "Portfolio share (%)"

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            display_topics[["topic", "count", "share"]].rename(
                columns={"topic": "Core Topic", "count": "Papers", "share": share_label}
            ),
            use_container_width=True, hide_index=True
        )
    with col2:
        fig_bar = px.bar(
            display_topics, x="share", y="topic", orientation="h",
            labels={"share": share_label, "topic": "Core Topic"},
            title=f"Core Topic Architecture — {active_timeframe_string}"
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("ℹ️ How to interpret this topic distribution"):
        if st.session_state.use_evolution_vector:
            st.markdown("""
**This is the Chasing Mode target, not the journal's ordinary topic distribution.**

- CATALEX first calculates the change in each topic's portfolio share between Epoch 1 and Epoch 2.
- Only **positive changes** are retained.
- Those positive changes are then normalized to sum to **100%**. The displayed percentage is therefore the topic's **share of all positive topic-share growth**.
- Example: **10.7% here does not mean the topic accounts for 10.7% of Epoch 2 papers**. It means the topic contributes 10.7% of the normalized positive-growth vector used to find journals aligned with the target's expanding areas.

**Use this view for:** understanding what CATALEX is actually chasing when Chasing Mode ranks competitors.
""")
        else:
            st.markdown("""
**This is the target journal's current structural topic profile.**

- **Papers** is the number of works assigned to each primary OpenAlex topic in the selected period.
- **Portfolio share (%)** is that topic's proportion of the target journal's works across its complete primary-topic profile.
- The displayed core topics are the strongest components used to discover plausible competitor journals.

**Use this view for:** understanding the target journal's present scope before interpreting the competitor ranking.
""")

    st.subheader("Extracting Market Competitors")
    with st.expander("ℹ️ What competitor extraction and similarity mean"):
        st.markdown("""
CATALEX first discovers journals that publish sufficiently often in the target's normal/static core topics. This established competitor universe does not change when Chasing Mode is switched on. In Chasing Mode, those competitors are then re-profiled for Epoch 2 and ranked against the target journal's normalized positive-growth vector.

- **Cosine Similarity** measures alignment of the two topic-share vectors: **1 = identical direction/profile shape; 0 = no alignment**. It does not mean that a score of 0.80 implies 80% of papers overlap.
- **TF-IDF Cosine** additionally down-weights topics common across many journals in the candidate pool, giving more influence to distinctive shared topics. Because those weights depend on the current candidate set, TF-IDF scores should not be treated as fixed values comparable across differently configured runs.
- Publication volume and similarity answer different questions: a much larger or smaller journal can still have high cosine similarity if its relative topic mix is similar.

**Use the ranking for:** finding journals with structurally similar scope, followed by editorial judgement about whether they are meaningful competitors.
""")
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

    # Apply any user-supplied journal metric before the expensive profile scans.
    metric_active = metric_df_raw is not None and metric_title_col and metric_value_col
    if metric_active:
        metric_min = parse_optional_bound(metric_min_text, "Minimum metric")
        metric_max = parse_optional_bound(metric_max_text, "Maximum metric")
        if metric_min is not None and metric_max is not None and metric_min > metric_max:
            raise ValueError("Minimum metric must not be greater than maximum metric.")
        title_lookup, issn_lookup, usable_metric_rows = prepare_metric_lookup(
            metric_df_raw, metric_title_col, metric_value_col, metric_issn_col, metric_duplicate_rule
        )
        with st.spinner(f"Matching candidate journals to {metric_value_col}..."):
            top_candidates = attach_user_metric(
                top_candidates, title_lookup, issn_lookup, user_api_key, user_mailto
            )
        matched_before_filter = int(top_candidates["User Metric"].notna().sum())
        metric_pass = pd.Series(True, index=top_candidates.index)
        if metric_min is not None:
            metric_pass &= top_candidates["User Metric"] >= metric_min
        if metric_max is not None:
            metric_pass &= top_candidates["User Metric"] <= metric_max
        if include_unmatched_metric:
            metric_pass |= top_candidates["User Metric"].isna()
        else:
            metric_pass &= top_candidates["User Metric"].notna()
        top_candidates = top_candidates[metric_pass].copy()
        st.info(
            f"Uploaded metric: **{metric_value_col}**. Matched {matched_before_filter} candidate journals; "
            f"{len(top_candidates)} remain after the selected metric range and unmatched-journal rule."
        )
        if top_candidates.empty:
            st.warning("No candidate journals remain after metric filtering.")
            st.stop()
    else:
        top_candidates["User Metric"] = np.nan
        top_candidates["Metric Match"] = "No metric file"

    candidate_profiles = {}
    st.markdown("##### 🔄 Processing Competitor Vector Profiles")

    # Competitor discovery always comes from the normal/static CATALEX scope.
    # Chasing Mode changes only the comparison: established competitors are
    # re-profiled in Epoch 2 and ranked against the target's positive-growth vector.
    competitor_profile_window = epoch_2 if (st.session_state.use_evolution_vector and "dynamic_growth_vector" in st.session_state) else FROM_DATE
    
    def fetch_profile_worker(source_id, display_name):
        try:
            profile_df = journal_topic_profile(source_id, date_value=competitor_profile_window, api_key=user_api_key, mailto=user_mailto)
            if profile_df.empty:
                return source_id, {}, "Empty topic profile"
            return source_id, dict(zip(profile_df["topic_id"], profile_df["share"])), None
        except Exception as exc:
            # Preserve the failure reason instead of silently assigning similarity 0.
            return source_id, {}, str(exc)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(fetch_profile_worker, row["source_id"], row["journal"]): row 
            for _, row in top_candidates.iterrows()
        }
        
        c_progress = st.progress(0, text="Initializing concurrent lookups...")
        completed = 0
        failed_profiles = []
        total_futures = len(futures)
        
        for future in as_completed(futures):
            s_id, profile_map, error_message = future.result()
            completed += 1
            if profile_map:
                candidate_profiles[s_id] = profile_map
            elif error_message:
                failed_profiles.append((s_id, error_message))
            c_progress.progress(completed / total_futures, text=f"Fetched profile tracking: {completed}/{total_futures}")
        c_progress.empty()

    if failed_profiles:
        st.warning(f"{len(failed_profiles)} candidate profiles could not be loaded and were excluded.")

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
    with st.expander("ℹ️ How to interpret the visual diagnostics"):
        if st.session_state.use_evolution_vector:
            st.markdown("""
**Chasing Mode mixes two intentionally different quantities in these diagnostics:**

- **TARGET row/trace:** normalized share of total positive topic-share growth.
- **Competitor rows/traces:** actual percentage of each competitor's portfolio assigned to those same topics.

For example, a target value of 10.7% and a competitor value of 14.7% do **not** mean the same thing and should not be subtracted. The visualization asks whether competitors have meaningful presence in the topics driving the target journal's expansion.

**Use the heatmap** to see which growing target fields are strongly represented in each competitor. **Use the radar** to inspect the overall shape of alignment across those fields.
""")
        else:
            st.markdown("""
In Standard Mode, the target and competitor values are directly comparable: every cell or radar point is the topic's **percentage share of that journal's portfolio**.

**Use the heatmap** for precise topic-by-journal comparisons. **Use the radar** for a quick visual impression of whether two journals have similarly shaped topic footprints.
""")

    viz_tab1, viz_tab2 = st.tabs(["📊 Scope Overlay Heatmap Matrix", "🕸️ Scope Overlap Radar Map"])

    id_to_name = dict(zip(top_candidates["source_id"], top_candidates["journal"]))
    id_to_name[target_id] = f"TARGET: {target['display_name']} ({active_timeframe_string})"

    df_heatmap = pd.DataFrame(matrix_visualization_data)
    valid_journals = list(top_candidates["source_id"].unique()) + [target_id]
    df_heatmap = df_heatmap[df_heatmap["Source_ID"].isin(valid_journals)]

    with viz_tab1:
        if st.session_state.use_evolution_vector:
            st.markdown("**Chasing Mode Comparison Matrix**")
            st.caption("Target row = normalized share of the target journal's total positive topic-share growth. Competitor rows = each topic's actual share of that competitor journal portfolio in Epoch 2. The units therefore differ between the target row and competitor rows; use this matrix as a directional diagnostic, not as a same-unit share comparison.")
        else:
            st.markdown(f"**Journal Portfolio Topic-Share Matrix ({active_timeframe_string})**")
            st.caption("All cells show the topic's percentage share of the corresponding journal portfolio for the selected standard time window.")
        pivot_hm = df_heatmap.pivot(index="Source_ID", columns="Topic", values="Share (%)").fillna(0)
        heatmap_order = [target_id] + list(top_candidates["source_id"].unique())
        pivot_hm = pivot_hm.reindex(heatmap_order)
        pivot_hm.index = [id_to_name.get(x, x) for x in pivot_hm.index]

        fig_hm = px.imshow(
            pivot_hm, text_auto=True, aspect="auto",
            labels=dict(x="Topic Fields", y="Journals", color="Value (%)"),
            color_continuous_scale="Viridis",
            title=(f"Growth Priorities vs Competitor Portfolio Shares ({active_timeframe_string})" if st.session_state.use_evolution_vector else f"Journal Portfolio Topic-Share Comparison ({active_timeframe_string})")
        )
        fig_hm.update_yaxes(type="category", dtick=1)
        if st.session_state.use_evolution_vector:
            fig_hm.update_traces(hovertemplate="Topic: %{x}<br>Journal / target: %{y}<br>Value: %{z:.1f}%<br><i>Target row = share of positive growth; competitor rows = portfolio share.</i><extra></extra>")
        else:
            fig_hm.update_traces(hovertemplate="Topic: %{x}<br>Journal: %{y}<br>Portfolio share: %{z:.1f}%<extra></extra>")
        fig_hm.update_layout(height=800, margin=dict(l=200, r=20, t=50, b=150))
        st.plotly_chart(fig_hm, use_container_width=True)

    with viz_tab2:
        if st.session_state.use_evolution_vector:
            st.markdown("**Growth-Vector Alignment Radar**")
            st.caption("Target trace = normalized positive-growth weights. Competitor traces = actual portfolio shares in the same topics. This is an alignment aid; the target and competitor traces do not represent identical units in Chasing Mode.")
        else:
            st.markdown(f"**Portfolio Fingerprint Alignment ({active_timeframe_string})**")
            st.caption("All traces show portfolio topic shares, allowing direct comparison of journal scope fingerprints.")
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
    ranking_columns = ["journal", "Similarity", "shared_target_topics", "works_in_target_topics"]
    ranking_rename = {
        "journal": "Journal Title Space", "Similarity": f"{SIM_METRIC}",
        "shared_target_topics": "Overlapping Core Topics",
        "works_in_target_topics": "Total Intersecting Publications"
    }
    if metric_active:
        ranking_columns += ["User Metric", "Metric Match"]
        ranking_rename["User Metric"] = metric_value_col
    st.dataframe(
        top_candidates[ranking_columns].rename(columns=ranking_rename),
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
