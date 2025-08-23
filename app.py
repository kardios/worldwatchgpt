import os
import streamlit as st
from openai import OpenAI

# Set page config
st.set_page_config(page_title="WorldWatch GPT", layout="wide")

# Load API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment variables.")
    st.stop()

client = OpenAI(api_key=api_key)

# Default prompt template
default_prompt = """TASK

Draft a 200-word write-up (+/- 15% length) on the perspective of <<PUT COUNTRY HERE>> on <<PUT TOPIC HERE>>.

STYLE & EVIDENCE

• Use British English, third-person, neutral tone, and dense informative paragraphs in prose (no bullet points).

• Begin each paragraph by presenting a claim, then supporting it with evidence (like quotes, data, or examples), and finally analysing that evidence to explain its relevance to the claim. Use no more than two paragraphs.

• Maintain a professional, slightly formal register without being overly academic.

• Acknowledge uncertainties, risks, and counter arguments explicitly.

• Cite authoritative sources (e.g., official government press releases, statistics, and white papers; UN voting records; reputable news sources) at the end.

• Do not overemphasise human-rights discourse unless it materially affects policy, security, or external relations considerations.
"""

# App title
st.title("🌍 WorldWatch GPT")
st.markdown("Generate concise, evidence-based country perspectives on global topics using GPT-5.")

# Sidebar controls
st.sidebar.header("Country & Topic Selection")
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    [
        "Japan", "Vietnam", "China", "United States", "United Kingdom",
        "Australia", "India", "Russia", "France", "Germany"
    ]
)

topic = st.sidebar.text_input(
    "Enter the topic to research",
    placeholder="e.g. US-China rivalry, Israeli-Palestinian conflict, AI regulations"
)

# Generate reports button
if st.sidebar.button("Generate Reports"):
    if not selected_countries:
        st.warning("Please select at least one country.")
    elif not topic.strip():
        st.warning("Please enter a topic to research.")
    else:
        for country in selected_countries:
            st.subheader(f"Perspective: {country}")

            # Prepare the final prompt
            prompt = default_prompt.replace("<<PUT COUNTRY HERE>>", country).replace("<<PUT TOPIC HERE>>", topic)

            # Call GPT-5 API
            with st.spinner(f"Generating perspective for {country}..."):
                try:
                    response = client.responses.create(
                        model="gpt-5-mini",
                        input=prompt,
                        reasoning={"effort": "low"},
                        text={"verbosity": "medium"},
                        tools=[{"type": "web_search"}],
                        tool_choice="auto"
                    )
                    report_text = response.output_text
                except Exception as e:
                    st.error(f"Error generating report for {country}: {str(e)}")
                    continue

            st.write(report_text)
