import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000")

# Every API endpoint except /health is key-gated. On Streamlit Community Cloud
# this comes from the app's secrets; locally it comes from the environment.
def _api_key() -> str:
    if key := os.environ.get("API_KEY"):
        return key
    try:
        return st.secrets.get("API_KEY", "")
    except Exception:
        # st.secrets raises rather than returning empty when no secrets file
        # exists at all, which is the normal case for a plain local run.
        return ""


AUTH_HEADERS = {"X-API-Key": _api_key()}

st.set_page_config(
    page_title="Text-to-SQL Guardrails",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background: #1A1D24;
    border: 1px solid #2A2D34;
    border-radius: 12px;
    padding: 14px;
}

.confidence-pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
}

.high {
    background: rgba(34,197,94,.15);
    color: #22C55E;
}

.medium {
    background: rgba(234,179,8,.15);
    color: #EAB308;
}

.low {
    background: rgba(239,68,68,.15);
    color: #EF4444;
}

.signal-label {
    display: flex;
    justify-content: space-between;
    font-size: .85rem;
    color: #B4B4B4;
    margin-bottom: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🛡️ Text-to-SQL Interface with Guardrails")
st.caption(
    "Natural language → SQL with guardrails, hallucination detection and confidence scoring."
)

tab_query, tab_schema, tab_metrics = st.tabs(["💬 Query", "🗂️ Schema Browser", "📊 Metrics"])


def confidence_pill(score: float) -> str:
    if score >= 0.7:
        cls, label = "high", "High"
    elif score >= 0.4:
        cls, label = "medium", "Medium"
    else:
        cls, label = "low", "Low"

    return f'<span class="confidence-pill {cls}">{label} Confidence • {score:.2f}</span>'


def signal_bar(name: str, value: float) -> None:
    st.markdown(
        f'<div class="signal-label"><span>{name.replace("_", " ").title()}</span>'
        f'<span>{value:.2f}</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(value, 0.0), 1.0))


# -----------------------------------------------------------------------------
# QUERY TAB
# -----------------------------------------------------------------------------
with tab_query:
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.container(border=True):
        question = st.text_input(
            "Ask a question about your database",
            placeholder="e.g. How many orders has each customer placed?",
        )
        run_clicked = st.button("▶ Run Query", type="primary")

    if run_clicked and question.strip():
        with st.spinner("Generating SQL → validating → executing → checking hallucinations..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/v1/query",
                    json={"question": question},
                    headers=AUTH_HEADERS,
                    timeout=120,
                )
                result = response.json()
            except requests.RequestException as e:
                st.error(f"Could not reach API:\n\n{e}")
                result = None

        if result:
            st.session_state.history.insert(0, {"question": question, "result": result})

            if not result.get("executed", False):
                with st.container(border=True):
                    st.error(
                        f"🚫 Blocked by Guardrail\n\n"
                        f"{result.get('blocked_reason') or result.get('error')}"
                    )
                    st.code(result.get("generated_sql", ""), language="sql")
            else:
                confidence = result.get("confidence", 0.0)
                left, right = st.columns([2, 1], gap="large")

                with left:
                    with st.container(border=True):
                        st.subheader("Generated SQL")
                        st.code(result["generated_sql"], language="sql")
                        st.caption("💡 " + result.get("explanation", ""))

                    with st.container(border=True):
                        st.subheader("Results")
                        rows = result.get("rows", [])
                        if rows:
                            st.dataframe(
                                pd.DataFrame(rows),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("Query executed successfully but returned zero rows.")
                        st.caption(
                            f"📄 {result.get('row_count', 0)} rows"
                            f"   •   "
                            f"⏱ {result.get('execution_time_ms', 0):.1f} ms"
                        )

                with right:
                    with st.container(border=True):
                        st.markdown(confidence_pill(confidence), unsafe_allow_html=True)
                        st.write("")

                        breakdown = result.get("confidence_breakdown", {})
                        for signal, value in breakdown.items():
                            signal_bar(signal, value)

                        issues = result.get("flagged_issues", [])
                        if issues:
                            st.write("")
                            st.warning(
                                "**Flagged Issues**\n\n"
                                + "\n".join(f"- {issue}" for issue in issues)
                            )

                        back_translated = result.get("back_translated_question")
                        if back_translated:
                            st.write("")
                            st.caption("🔄 System Interpretation:\n\n" + back_translated)

    if st.session_state.history:
        st.divider()
        st.subheader("📜 Recent Queries")

        for item in st.session_state.history[1:6]:
            confidence = item["result"].get("confidence")
            conf = f"{confidence:.2f}" if confidence is not None else "N/A"

            with st.expander(f"{item['question']}   •   {conf}"):
                st.code(item["result"].get("generated_sql", ""), language="sql")

# -----------------------------------------------------------------------------
# SCHEMA TAB
# -----------------------------------------------------------------------------
with tab_schema:
    st.subheader("🗂️ Live Database Schema")
    st.caption("Introspected directly from PostgreSQL using SQLAlchemy.")

    if st.button("🔄 Refresh Schema"):
        st.rerun()

    try:
        response = requests.get(f"{API_BASE_URL}/schema", headers=AUTH_HEADERS, timeout=30)
        tables = response.json()

        cols = st.columns(2)
        for i, table in enumerate(tables):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### 📋 {table['name']}")
                    st.dataframe(
                        pd.DataFrame(table["columns"]),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if table.get("foreign_keys"):
                        st.write("**Foreign Keys**")
                        for fk in table["foreign_keys"]:
                            st.caption(
                                f"🔗 `{table['name']}.{fk['column']}` → "
                                f"`{fk['references_table']}.{fk['references_column']}`"
                            )
    except requests.RequestException as e:
        st.error(f"Could not load schema:\n\n{e}")

# -----------------------------------------------------------------------------
# METRICS TAB
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("📊 Live System Metrics")
    st.caption("These counters are stored in memory and reset whenever the API restarts.")

    if st.button("🔄 Refresh Metrics"):
        st.rerun()

    try:
        response = requests.get(f"{API_BASE_URL}/v1/metrics", headers=AUTH_HEADERS, timeout=30)
        metrics = response.json()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", metrics.get("total_queries", 0))
        c2.metric("Blocked Queries", metrics.get("blocked_queries", 0))
        c3.metric("Block Rate", f"{(metrics.get('block_rate') or 0):.1%}")
        c4.metric("Average Confidence", f"{metrics.get('avg_confidence') or 0:.2f}")

        st.write("")
        left, right = st.columns(2)

        with left:
            st.metric(
                "Flagged Issue Rate",
                f"{(metrics.get('flagged_issue_rate') or 0):.1%}",
            )

        with right:
            avg_conf = metrics.get("avg_confidence")

            if avg_conf is None:
                st.info("No successful queries have been executed yet.")
            elif avg_conf >= 0.7:
                st.success("🟢 System confidence is healthy")
            elif avg_conf >= 0.4:
                st.warning("🟡 Moderate confidence")
            else:
                st.error("🔴 Low confidence")

    except requests.RequestException as e:
        st.error(f"Could not load metrics:\n\n{e}")