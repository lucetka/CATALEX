import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    label="Minimum Cosine Similarity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Filters competitors by cosine similarity of their full OpenAlex topic profiles. 0 = no alignment; 1 = identical proportional topic profile."
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

st.title("🐈CATALEX")

st.caption(
    "Cosine Alignment of Topics via OpenAlex - dynamic scope-based only "
    "journal competitor finder using OpenAlex true global profiles. "
    f"Target journal: **{TARGET_JOURNAL}**"
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
    data = oa_get(
        "sources",
        {
            "search": target_name,
            "per-page": 10
        },
        api_key=api_key,
        mailto=mailto
    )

    results = data.get("results", [])

    for x in results:
        if x.get("display_name", "").strip().lower() == target_name.lower():
            return x

    if results:
        return results[0]

    raise ValueError(
        f"Could not find {target_name} in OpenAlex."
    )


@st.cache_data(show_spinner=False)
def journal_topic_profile(source_id, from_date, api_key="", mailto=""):
    data = oa_get(
        "works",
        {
            "filter":
                f"primary_location.source.id:{source_id},"
                f"from_publication_date:{from_date}",
            "group_by": "primary_topic.id",
        },
        api_key=api_key,
        mailto=mailto
    )

    groups = data.get("group_by", [])

    if not groups:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic",
                "count",
                "share"
            ]
        )

    df = pd.DataFrame([
        {
            "topic_id": x["key"].split("/")[-1],
            "topic": x["key_display_name"],
            "count": x["count"],
        }
        for x in groups
    ])

    total = df["count"].sum()

    df["share"] = (
        df["count"] / total
        if total else 0
    )

    return df.sort_values(
        "count",
        ascending=False
    )


@st.cache_data(show_spinner=False)
def journals_for_topic(
    topic_id,
    from_date,
    min_works,
    max_journals,
    api_key="",
    mailto=""
):
    data = oa_get(
        "works",
        {
            "filter":
                f"primary_topic.id:{topic_id},"
                f"from_publication_date:{from_date},"
                f"primary_location.source.type:journal",
            "group_by":
                "primary_location.source.id",
        },
        api_key=api_key,
        mailto=mailto
    )

    rows = []

    for x in data.get("group_by", []):
        if x["count"] >= min_works:
            rows.append({
                "source_id":
                    x["key"].split("/")[-1],
                "journal":
                    x["key_display_name"],
                "topic_count":
                    x["count"],
            })

    return pd.DataFrame(rows).head(
        max_journals
    )


def build_candidate_set(
    target_topics,
    from_date,
    baseline_min_papers,
    max_journals,
    api_key="",
    mailto=""
):
    candidate_frames = []

    progress = st.progress(
        0,
        text="Finding journals sharing target topics..."
    )

    for i, row in target_topics.iterrows():

        dynamic_min_works = max(
            baseline_min_papers,
            int(row["count"] * 0.10)
        )

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

        progress.progress(
            (i + 1) / len(target_topics),
            text=f"Scanning topic {i + 1}/{len(target_topics)}"
        )

    progress.empty()

    if not candidate_frames:
        return pd.DataFrame()

    df_all = pd.concat(
        candidate_frames,
        ignore_index=True
    )

    summary = df_all.groupby(
        ["source_id", "journal"]
    ).agg(
        shared_target_topics=(
            "topic_id",
            "nunique"
        ),
        works_in_target_topics=(
            "topic_count",
            "sum"
        )
    ).reset_index()

    return summary.sort_values(
        [
            "shared_target_topics",
            "works_in_target_topics"
        ],
        ascending=False
    )


def cosine_similarity(profile_a, profile_b):

    topics = (
        set(profile_a.keys())
        | set(profile_b.keys())
    )

    a = np.array([
        profile_a.get(t, 0.0)
        for t in topics
    ])

    b = np.array([
        profile_b.get(t, 0.0)
        for t in topics
    ])

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    return (
        float(np.dot(a, b) / denom)
        if denom > 0
        else 0.0
    )


# ---------------------------------------------------------
# MAIN PIPELINE RUNTIME
# ---------------------------------------------------------

try:

    if not TARGET_JOURNAL.strip():
        st.warning(
            "Please type a valid target journal name in the sidebar."
        )
        st.stop()

    # Resolve target journal
    target = find_target_source(
        TARGET_JOURNAL,
        api_key=user_api_key,
        mailto=user_mailto
    )

    target_id = target["id"].split("/")[-1]

    st.success(
        f"Target resolved as: "
        f"{target['display_name']} "
        f"({target_id})"
    )

    with st.spinner(
        f"Getting true topic profile for "
        f"{target['display_name']}..."
    ):
        full_target_profile = journal_topic_profile(
            target_id,
            from_date=FROM_DATE,
            api_key=user_api_key,
            mailto=user_mailto
        )

    if full_target_profile.empty:
        st.error(
            "No topic data found for this journal "
            "in the chosen date range."
        )
        st.stop()


    # -----------------------------------------------------
    # Dynamic target topic selection
    # -----------------------------------------------------

    full_target_profile["cumulative_share"] = (
        full_target_profile["share"].cumsum()
    )

    threshold_df = full_target_profile[
        full_target_profile[
            "cumulative_share"
        ] <= 0.75
    ]

    dynamic_n = max(
        5,
        min(
            len(threshold_df) + 1,
            20
        )
    )

    target_topics = (
        full_target_profile
        .head(dynamic_n)
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # TARGET PROFILE DISPLAY
    # -----------------------------------------------------

    st.subheader(
        f"{target['display_name']} "
        "Structural Field Distribution"
    )

    display_topics = target_topics.copy()

    display_topics["share"] = (
        display_topics["share"] * 100
    ).round(1)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.dataframe(
            display_topics[
                [
                    "topic",
                    "count",
                    "share"
                ]
            ].rename(
                columns={
                    "topic": "Core Topic",
                    "count": "Papers",
                    "share": "%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    with col2:

        # Plotly version of the original bar chart
        # so that it can be downloaded as HTML.
        fig_bar = px.bar(
            display_topics,
            x="share",
            y="topic",
            orientation="h",
            labels={
                "share": "Share (%)",
                "topic": "Core Topic"
            },
            title="Core Topic Distribution"
        )

        fig_bar.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            },
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

        st.download_button(
            label="⬇️ Download bar chart as HTML",
            data=plotly_html_bytes(fig_bar),
            file_name="CATALEX_topic_distribution.html",
            mime="text/html",
            key="download_bar_html"
        )


    # -----------------------------------------------------
    # CORE CANDIDATE PROCESSING
    # -----------------------------------------------------

    st.subheader(
        "Extracting Market Competitors"
    )

    candidates = build_candidate_set(
        target_topics,
        from_date=FROM_DATE,
        baseline_min_papers=min_papers,
        max_journals=JOURNALS_PER_TOPIC,
        api_key=user_api_key,
        mailto=user_mailto
    )

    if candidates.empty:
        st.warning(
            "No candidate competitor journals found "
            "matching structural parameters."
        )
        st.stop()

    # Remove target journal itself
    candidates = candidates[
        candidates["source_id"] != target_id
    ]


    # -----------------------------------------------------
    # WHITELIST OVERRIDES
    # -----------------------------------------------------

    is_whitelisted = (
        candidates["journal"]
        .isin(whitelist_journals)
    )

    whitelisted_pool = candidates[
        is_whitelisted
    ].copy()

    standard_pool = candidates[
        ~is_whitelisted
    ].copy()

    min_shared_topics = max(
        2,
        int(
            len(target_topics) * 0.25
        )
    )

    filtered_standard = standard_pool[
        standard_pool[
            "shared_target_topics"
        ] >= min_shared_topics
    ].copy()

    top_candidates = pd.concat(
        [
            whitelisted_pool,
            filtered_standard
        ],
        ignore_index=True
    ).drop_duplicates(
        subset=["source_id"]
    )

    top_candidates = (
        top_candidates.sort_values(
            [
                "shared_target_topics",
                "works_in_target_topics"
            ],
            ascending=False
        )
    )

    if len(top_candidates) > max_candidates:
        top_candidates = (
            top_candidates
            .head(max_candidates)
        )


    # -----------------------------------------------------
    # VECTOR ARRAYS
    # -----------------------------------------------------

    target_vector_map = dict(
        zip(
            full_target_profile["topic_id"],
            full_target_profile["share"]
        )
    )

    similarities = []
    matrix_visualization_data = []


    # Target coordinates for heatmap
    for _, t_row in target_topics.iterrows():

        matrix_visualization_data.append({
            "Source_ID":
                target_id,
            "Topic":
                t_row["topic"],
            "Share (%)":
                round(
                    target_vector_map.get(
                        t_row["topic_id"],
                        0.0
                    ) * 100,
                    1
                )
        })


    progress = st.progress(
        0,
        text=(
            "Analyzing true candidate profiles "
            "and running cosine vectors..."
        )
    )

    total_cand = len(top_candidates)

    for n, (_, row) in enumerate(
        top_candidates.iterrows()
    ):

        cand_profile_df = (
            journal_topic_profile(
                row["source_id"],
                from_date=FROM_DATE,
                api_key=user_api_key,
                mailto=user_mailto
            )
        )

        cand_vector_map = dict(
            zip(
                cand_profile_df["topic_id"],
                cand_profile_df["share"]
            )
        )

        sim = cosine_similarity(
            target_vector_map,
            cand_vector_map
        )

        similarities.append(sim)

        for _, t_row in target_topics.iterrows():

            matrix_visualization_data.append({
                "Source_ID":
                    row["source_id"],
                "Topic":
                    t_row["topic"],
                "Share (%)":
                    round(
                        cand_vector_map.get(
                            t_row["topic_id"],
                            0.0
                        ) * 100,
                        1
                    )
            })

        progress.progress(
            (n + 1) / total_cand,
            text=(
                f"Processing profile "
                f"{n + 1}/{total_cand}"
            )
        )

    progress.empty()


    top_candidates["Similarity"] = np.array(similarities).round(3)


    # -----------------------------------------------------
    # APPLY SIMILARITY FILTER
    # -----------------------------------------------------

    passed_filter = (
        (
            top_candidates["Similarity"]
            >= min_overlap_pct
        )
        |
        (
            top_candidates["journal"]
            .isin(whitelist_journals)
        )
    )

    top_candidates = top_candidates[
        passed_filter
    ]

    top_candidates = (
        top_candidates
        .sort_values(
            "Similarity",
            ascending=False
        )
        .head(30)
    )


    # -----------------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------------

    st.subheader(
        "Visual Scope Diagnostics"
    )

    viz_tab1, viz_tab2 = st.tabs([
        "📊 Scope Overlay Heatmap",
        "🕸️ Scope Overlap Radar Map"
    ])

    id_to_name = dict(
        zip(
            top_candidates["source_id"],
            top_candidates["journal"]
        )
    )

    id_to_name[target_id] = (
        f"TARGET: {target['display_name']}"
    )

    df_heatmap = pd.DataFrame(
        matrix_visualization_data
    )

    valid_journals = (
        list(
            top_candidates[
                "source_id"
            ].unique()
        )
        + [target_id]
    )

    df_heatmap = df_heatmap[
        df_heatmap[
            "Source_ID"
        ].isin(valid_journals)
    ]


    # -----------------------------------------------------
    # HEATMAP
    # -----------------------------------------------------

    with viz_tab1:

        st.markdown(
            "**True Scope Distribution Matrix:** "
            "Explicitly tracks concentration variables "
            "relative to core focus domains."
        )

        pivot_hm = df_heatmap.pivot(
            index="Source_ID",
            columns="Topic",
            values="Share (%)"
        ).fillna(0)

        heatmap_order = (
            [target_id]
            + list(
                top_candidates[
                    "source_id"
                ].unique()
            )
        )

        pivot_hm = pivot_hm.reindex(
            heatmap_order
        )

        pivot_hm.index = [
            id_to_name.get(x, x)
            for x in pivot_hm.index
        ]

        fig_hm = px.imshow(
            pivot_hm,
            text_auto=True,
            aspect="auto",
            labels=dict(
                x="Topic Fields",
                y="Competitor Journals",
                color="Global Share %"
            ),
            color_continuous_scale="Viridis"
        )

        fig_hm.update_yaxes(
            type="category",
            dtick=1
        )

        fig_hm.update_layout(
            height=800,
            margin=dict(
                l=200,
                r=20,
                t=20,
                b=150
            )
        )

        st.plotly_chart(
            fig_hm,
            use_container_width=True
        )

        st.download_button(
            label="⬇️ Download heatmap as HTML",
            data=plotly_html_bytes(fig_hm),
            file_name="CATALEX_scope_heatmap.html",
            mime="text/html",
            key="download_heatmap_html"
        )


    # -----------------------------------------------------
    # RADAR
    # -----------------------------------------------------

    with viz_tab2:

        st.markdown(
            "**Fingerprint alignment shapes:** "
            "Select competitors to trace architectural "
            "alignment directly against baseline shapes."
        )

        journal_options = list(
            top_candidates[
                "source_id"
            ].unique()
        )

        selected_ids = st.multiselect(
            "Select journals to trace:",
            options=journal_options,
            default=(
                journal_options[:2]
                if len(journal_options) >= 2
                else journal_options
            ),
            format_func=lambda x:
                id_to_name.get(x, x)
        )

        fig_radar = go.Figure()

        fig_radar.add_trace(
            go.Scatterpolar(
                r=(
                    target_topics["share"]
                    * 100
                ),
                theta=target_topics[
                    "topic"
                ],
                fill="toself",
                fillcolor=(
                    "rgba(26, 26, 26, 0.15)"
                ),
                name=(
                    f"TARGET: "
                    f"{target['display_name']}"
                ),
                line=dict(
                    color="#1A1A1A",
                    width=4,
                    dash="solid"
                ),
                marker=dict(
                    size=8,
                    color="#1A1A1A",
                    symbol="circle"
                )
            )
        )

        for s_id in selected_ids:

            sub_sj = df_heatmap[
                df_heatmap[
                    "Source_ID"
                ] == s_id
            ].copy()

            r_values = []

            for topic_name in (
                target_topics["topic"]
            ):

                match = sub_sj[
                    sub_sj[
                        "Topic"
                    ] == topic_name
                ]

                val = (
                    float(
                        match[
                            "Share (%)"
                        ].values[0]
                    )
                    if not match.empty
                    else 0.0
                )

                r_values.append(val)

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=target_topics[
                        "topic"
                    ],
                    fill="toself",
                    name=id_to_name.get(
                        s_id,
                        s_id
                    )
                )
            )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[
                        0,
                        max(
                            target_topics[
                                "share"
                            ] * 100
                        ) + 5
                    ]
                )
            ),
            showlegend=True,
            height=600
        )

        st.plotly_chart(
            fig_radar,
            use_container_width=True
        )

        st.download_button(
            label="⬇️ Download radar as HTML",
            data=plotly_html_bytes(
                fig_radar
            ),
            file_name="CATALEX_scope_radar.html",
            mime="text/html",
            key="download_radar_html"
        )


    # -----------------------------------------------------
    # COMPETITOR RANKING
    # -----------------------------------------------------

    st.subheader(
        "Competitor Ranking Summary Matrix"
    )

    st.dataframe(
        top_candidates[
            [
                "journal",
                "Similarity",
                "shared_target_topics",
                "works_in_target_topics"
            ]
        ].rename(
            columns={
                "journal":
                    "Journal Title Space",
                "Similarity":
                    "Cosine Similarity",
                "shared_target_topics":
                    "Overlapping Core Topics",
                "works_in_target_topics":
                    "Total Intersecting Publications"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # BOOLEAN QUERY STRING GENERATOR
    # -----------------------------------------------------

    st.subheader(
        "🔎 Competitor Query String Generator"
    )

    st.caption(
        "Generate a Boolean OR query from the "
        "highest-ranked identified competitor journals."
    )

    if not top_candidates.empty:

        max_query_journals = len(
            top_candidates
        )

        top_x = st.number_input(
            "Number of top competitors to include",
            min_value=1,
            max_value=max_query_journals,
            value=min(
                3,
                max_query_journals
            ),
            step=1
        )

        query_journals = (
            top_candidates
            .head(int(top_x))[
                "journal"
            ]
            .tolist()
        )

        boolean_query = (
            "("
            + " OR ".join(
                f'"{journal}"'
                for journal
                in query_journals
            )
            + ")"
        )

        st.text_area(
            "Generated query",
            value=boolean_query,
            height=100
        )

        st.download_button(
            label="⬇️ Download query as TXT",
            data=boolean_query,
            file_name=(
                "CATALEX_competitor_query.txt"
            ),
            mime="text/plain",
            key="download_query_txt"
        )


except Exception as e:

    st.error(
        f"Pipeline Runtime Exception: "
        f"{str(e)}"
    )
