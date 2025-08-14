import os
import io
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.express as px
import streamlit as st
import pycountry
from openai import OpenAI

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Country Briefings — GPT-5", page_icon="🌍", layout="wide")

# -----------------------
# API Client
# -----------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.info("Set your OpenAI API key in Streamlit secrets or the OPENAI_API_KEY env var.")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def all_country_names():
    # Build a list of country names understood by Plotly and general audiences
    names = sorted({c.name for c in pycountry.countries})
    # Add common synonyms that pycountry sometimes misses
    extras = ["Hong Kong", "Macau", "Kosovo"]
    for e in extras:
        if e not in names:
            names.append(e)
    return names

@st.cache_data(show_spinner=False)
def name_to_iso2(name: str) -> str | None:
    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_2
    except Exception:
        # simple fixes
        fixes = {
            "Vietnam": "VN",
            "Laos": "LA",
            "Bolivia": "BO",
            "Côte d’Ivoire": "CI",
            "Cote d'Ivoire": "CI",
            "Hong Kong": "HK",
            "Macau": "MO",
            "Kosovo": "XK",
            "Russia": "RU",
            "Syria": "SY",
            "South Korea": "KR",
            "North Korea": "KP",
            "Taiwan": "TW",
        }
        return fixes.get(name)

def render_sources_from_annotations(resp):
    sources = []
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for block in item.content or []:
                    if isinstance(block, dict):
                        anns = block.get("annotations") or []
                        for a in anns:
                            if a.get("type") == "url_citation" and a.get("url"):
                                title = a.get("title") or a["url"]
                                sources.append((title, a["url"]))
    except Exception:
        pass
    # dedupe by URL
    seen, uniq = set(), []
    for t,u in sources:
        if u not in seen:
            uniq.append((t,u)); seen.add(u)
    return uniq

def assemble_prompt(base_prompt: str, country: str) -> str:
    return base_prompt.replace("<<PUT COUNTRY HERE>>", country)

def build_tools(iso2: str, ctx_size: str):
    tool = {
        "type": "web_search_preview",
        "search_context_size": ctx_size,
    }
    if iso2:
        tool["user_location"] = {"type": "approximate", "country": iso2}
    return [tool]

def call_gpt5_report(
    client: OpenAI,
    country: str,
    iso2: str | None,
    base_prompt: str,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    search_context_size: str,
    require_search: bool,
    prev_id: str | None = None,
    preambles: bool = False,
):
    prompt_text = assemble_prompt(base_prompt, country)

    kwargs = dict(
        model=model,
        input=prompt_text,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
        tools=build_tools(iso2, search_context_size),
    )
    if require_search:
        kwargs["tool_choice"] = {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "web_search_preview"}],
        }
    if prev_id:
        kwargs["previous_response_id"] = prev_id
    if preambles:
        kwargs["system"] = "Before you call a tool, explain briefly why you are calling it."

    resp = client.responses.create(**kwargs)

    # Prefer convenience output_text
    output_text = getattr(resp, "output_text", "") or ""
    if not output_text:
        # reconstruct from message blocks if needed
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for block in item.content or []:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        output_text += block.get("text", "")
    sources = render_sources_from_annotations(resp)
    return resp.id, output_text, sources

def export_markdown(reports: list[dict]):
    # Combine to one Markdown file
    lines = []
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines.append(f"# Country Briefings — {now}\n")
    for r in reports:
        lines.append(f"## {r['country']}")
        lines.append("")
        lines.append(r["text"] or "_No content returned._")
        if r["sources"]:
            lines.append("\n**Sources**")
            for t,u in r["sources"]:
                lines.append(f"- [{t}]({u})")
        lines.append("\n---\n")
    md = "\n".join(lines)
    return md.encode("utf-8")

# -----------------------
# Sidebar Controls
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["gpt-5", "gpt-5-mini"], index=0)
    reasoning_effort = st.selectbox("Reasoning effort", ["minimal", "low", "medium", "high"], index=3)
    verbosity = st.selectbox("Verbosity", ["low", "medium", "high"], index=2)
    search_context_size = st.selectbox("Search context size", ["low", "medium", "high"], index=2)
    require_search = st.toggle("Require web search", value=True,
                               help="When ON, the model must use web_search_preview for fresh, cited information.")
    preambles = st.toggle("Explain tool calls (preambles)", value=False)
    max_workers = st.slider("Parallel countries (batch size)", 1, 12, 6)

# -----------------------
# Main Layout
# -----------------------
st.title("🌍 Country Briefings (GPT-5 Web Search + Reasoning)")
st.write("Generate **comprehensive political & economic reports** per country using GPT-5 with required web search and inline citations.")

# Map + multiselect
countries = all_country_names()

col_map, col_sel = st.columns([2, 1])
with col_map:
    df = pd.DataFrame({"country": countries, "value": 1})
    fig = px.choropleth(df, locations="country", locationmode="country names",
                        color="value", color_continuous_scale="Blues", labels={"value": ""},
                        title="World map (select countries from the list on the right)")
    fig.update_traces(hovertemplate="%{location}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=50, b=0), height=520)
    st.plotly_chart(fig, use_container_width=True)

with col_sel:
    st.subheader("Select countries")
    selected_countries = st.multiselect("Type to search and select multiple countries", countries, default=["Japan"])
    st.caption("Tip: Use the search box to quickly find countries.")

st.divider()
st.subheader("Report prompt")
default_prompt = """Prepare a comprehensive and detailed political report (approximately 1000 - 1200 words) on the current political and economic situation of <<PUT COUNTRY HERE>>. 

STYLE & EVIDENCE

• Use British English, third-person, neutral tone, and dense informative paragraphs (no bullet lists in the final report).

• Where possible, begin each paragraph by presenting a claim, then supporting it with evidence (like quotes, data, or examples), and finally analyzing that evidence to explain its relevance to the claim.

•  Maintain a professional, slightly formal register without being overly academic. Use third-person perspective. 

• Acknowledge uncertainties, risks, and counter arguments explicitly.

• Cite authoritative sources inline (e.g., World Bank, IMF, ADB, UN agencies, national statistics offices, central bank, finance/plan ministries, credible think-tanks; reputable media for recent events).

• If official data are missing, triangulate using credible estimates or proxies and state limitations briefly.

• Do not overemphasise human-rights discourse unless it materially affects policy, security, or external relations.

STRUCTURE 

The report should be structured as follows: 

1. Overview (one paragraph) : A succinct one-paragraph executive summary synthesising key political and economic developments, major challenges, and international dynamics. 

2. Political Developments (allocate ~50–60% of the report): State the political priorities and preoccupations of the current government. Include (where relevant) information on: governing coalition composition and dynamics; Dynamics among key leaders; major socio-politics and economic policy priorities and reforms; security and internal stability challenges; international preoccupations; and other domestic and international preoccupations. Organise into thematic sub-sections where relevant. 

3. Economic Developments: Present the country's latest GDP growth figures with year-on-year comparisons. Mention growth composition and sectoral drivers; inflation trajectory and underlying factors; labour market conditions. Identify principal economic opportunities and structural challenges. 

4. International Relations: Examine the country's foreign policy orientation and strategic priorities. Analyse relationships with: ASEAN and regional integration initiatives; immediate neighbours and border dynamics; major powers (China, United States, Japan); development partners and investors. Give particular attention to economic and security cooperation within Southeast Asia. Present all bilateral relationships objectively based on factual developments and official positions. 

5. There is no need for any concluding or wrap-up paragraph at the end of the report. 

DELIVERABLE

A single, coherent report with the above structure; dense paragraphs; precise dates; balanced treatment; and 8–12 citations to authoritative sources spread across sections.
"""
prompt_text = st.text_area("Customise prompt (per-country placeholder: <<PUT COUNTRY HERE>>)", default_prompt, height=380)

# Session state for previous_response_id per country (optional performance boost)
if "prev_ids" not in st.session_state:
    st.session_state.prev_ids = {}

# Generate Button
generate = st.button("🔎 Generate reports", use_container_width=True, type="primary", disabled=not selected_countries)

results_container = st.container()

if generate and selected_countries:
    st.toast("Generating reports…", icon="🛰️")
    # Parallel calls
    reports = []
    futures = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for country in selected_countries:
            iso2 = name_to_iso2(country) or ""
            prev_id = st.session_state.prev_ids.get(country)
            futures.append(pool.submit(
                call_gpt5_report,
                client, country, iso2, prompt_text, model, reasoning_effort, verbosity,
                search_context_size, require_search, prev_id, preambles
            ))
        for future in as_completed(futures):
            try:
                resp_id, text, sources = future.result()
                # Find country by searching in text if needed (best to map futures -> country index,
                # but for simplicity we reconstruct by order of completion with country extracted below)
                # We'll store as a dict; in UI, we show cards for each country in selected order.
                reports.append({"id": resp_id, "text": text, "sources": sources})
            except Exception as e:
                reports.append({"id": None, "text": f"_Error: {e}_", "sources": []})

    # Map back to countries (simple alignment by length; for exact mapping, we rerun order)
    # Here, we align by index to selected_countries if possible.
    aligned = []
    for i, country in enumerate(selected_countries):
        data = reports[i] if i < len(reports) else {"id": None, "text": "_No content._", "sources": []}
        st.session_state.prev_ids[country] = data["id"]
        aligned.append({"country": country, **data})

    elapsed = time.time() - start
    st.caption(f"Finished in {elapsed:.1f}s")

    # Render results
    tabs = st.tabs([c["country"] for c in aligned])
    for tab, r in zip(tabs, aligned):
        with tab:
            st.markdown(f"### {r['country']} — Report")
            st.write(r["text"] or "_No content returned._")
            if r["sources"]:
                st.markdown("**Sources**")
                for title, url in r["sources"]:
                    st.markdown(f"- [{title}]({url})")

    # Export
    md_bytes = export_markdown(aligned)
    st.download_button("⬇️ Download all as Markdown", data=md_bytes, file_name="country_briefings.md", mime="text/markdown")