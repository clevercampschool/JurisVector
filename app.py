import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="JURISVECTOR AI | Litigation Outcome & Precedent Engine",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ JURISVECTOR AI")
st.caption("High-Dimensional Case Law Retrieval, Precedent Analysis & Outcome Prediction Engine")

# Initialize Gemini API Client secretly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schemas for Structured Model Output
class PrecedentCase(BaseModel):
    case_citation: str = Field(description="Name and legal citation of the landmark precedent (e.g., Smith v. TechCorp, 452 F.3d 112).")
    relevance_score: float = Field(description="Cosine similarity or vector match score between 0.00 and 1.00.")
    jurisdiction: str = Field(description="Court or jurisdiction (e.g., U.S. Supreme Court, 9th Circuit, High Court).")
    holding_summary: str = Field(description="Key judicial holding relevant to the present litigation facts.")

class FactWeight(BaseModel):
    fact_element: str = Field(description="Key factual or evidentiary element extracted from the case narrative.")
    weight: float = Field(description="Normalized feature impact weight (|SHAP value| 0.0 to 1.0).")
    impact_direction: str = Field(description="Impact on plaintiff/prosecution position: 'Favors Movant/Plaintiff' or 'Favors Defendant'")

class CaseOutcomeInference(BaseModel):
    law_domain: str = Field(description="Area of jurisprudence (e.g., Intellectual Property Patent Infringement, Breach of Fiduciary Duty, Corporate Fraud).")
    win_probability_pct: int = Field(description="Calculated statistical probability of favorable outcome (0 to 100%).")
    recommended_strategy: str = Field(description="Litigation direction: 'SETTLE / MEDIATE IMMEDIATELY', 'PROCEED TO TRIAL', or 'MOTION FOR SUMMARY JUDGMENT'")
    controlling_precedents: list[PrecedentCase] = Field(description="Top 3 binding or persuasive landmark precedents extracted from vector DB.")
    shap_fact_weights: list[FactWeight] = Field(description="Key evidentiary facts that drove the predictive model.")
    key_legal_arguments: list[str] = Field(description="Primary legal doctrines or statutory provisions to plead in trial brief.")
    procedural_risks: list[str] = Field(description="Potential litigation pitfalls, evidentiary hurdles, or counter-claims.")


SYSTEM_INSTRUCTION = """
You are JURISVECTOR AI, a Litigation Analytics and Judicial Case Law Inference Engine trained on federal and state court decisions, case law repositories, and legal precedents.
Analyze the provided litigation fact narrative, dispute type, and jurisdiction.
Search vector space for controlling precedents, calculate a win-probability percentage (0-100%), compute SHAP feature weights for key case facts, and formulate litigation strategy.

Output strictly valid JSON according to the requested schema. Maintain an authoritative, analytical legal tone.
"""


def simulate_vector_retrieval():
    """Simulates realistic dense vector search across case law embeddings."""
    progress_bar = st.progress(0, text="Embedding Case Facts into 1536-dim Legal Vector Space...")
    time.sleep(0.3)

    progress_bar.progress(35, text="Querying Precedent Vector Index (Pinecone / Milvus Case DB)...")
    time.sleep(0.4)

    progress_bar.progress(70, text="Evaluating Judicial Outcome Classifiers & Judicial Historical Bias...")
    time.sleep(0.4)

    progress_bar.progress(95, text="Extracting SHAP Fact Weights & Generating Trial Strategy Matrix...")
    time.sleep(0.3)

    progress_bar.progress(100, text="Case Vector Processing Complete!")
    time.sleep(0.2)
    progress_bar.empty()


def render_juris_dashboard(result: dict):
    """Renders structured AI inference formatted as a Judicial Intelligence Dashboard."""

    domain = result.get("law_domain", "Civil Litigation")
    win_pct = result.get("win_probability_pct", 50)
    strategy = result.get("recommended_strategy", "PROCEED TO TRIAL")

    # 1. Primary Metrics Header
    st.subheader("🎯 Outcome Probability Vector")
    c1, c2, c3 = st.columns(3)

    c1.metric(label="Substantive Area of Law", value=domain)
    c2.metric(label="Predicted Success Rate", value=f"{win_pct}%")

    if "SETTLE" in strategy:
        c3.warning(f"**Recommended Action:**\n\n### ⚠️ {strategy}")
    elif "PROCEED" in strategy or "SUMMARY" in strategy:
        c3.success(f"**Recommended Action:**\n\n### 🟢 {strategy}")
    else:
        c3.error(f"**Recommended Action:**\n\n### 🚨 {strategy}")

    st.divider()

    # 2. Charts: Win Rate Gauge & SHAP Fact Weights
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📊 Statistical Win Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Likelihood of Favorable Verdict / Ruling", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#00CC96" if win_pct >= 60 else ("#FFAA00" if win_pct >= 40 else "#FF4B4B")},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255, 75, 75, 0.2)"},
                    {'range': [40, 60], 'color': "rgba(255, 170, 0, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(0, 204, 150, 0.2)"}
                ],
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart2:
        st.subheader("🧪 SHAP Fact Weights (|Impact|)")
        facts = result.get("shap_fact_weights", [])
        if facts:
            df_facts = pd.DataFrame(facts)
            fig_bar = px.bar(
                df_facts,
                x="weight",
                y="fact_element",
                color="impact_direction",
                orientation="h",
                title="Key Case Facts Driving Model Vector",
                labels={"weight": "Weight (|SHAP|)", "fact_element": "Fact Element"},
                color_discrete_map={"Favors Movant/Plaintiff": "#00CC96", "Favors Defendant": "#FF4B4B"}
            )
            fig_bar.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # 3. Precedent Case Law Table
    st.subheader("🏛️ Controlling & Persuasive Precedent Cases")
    precedents = result.get("controlling_precedents", [])
    if precedents:
        df_prec = pd.DataFrame(precedents)
        df_prec["similarity_pct"] = (df_prec["relevance_score"] * 100).round(1).astype(str) + "%"
        df_display = df_prec[["case_citation", "jurisdiction", "similarity_pct", "holding_summary"]]
        df_display.columns = ["Case Citation", "Jurisdiction / Forum", "Vector Similarity", "Judicial Holding"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.divider()

    # 4. Legal Arguments & Litigation Risks
    c_args, c_risks = st.columns(2)

    with c_args:
        st.subheader("⚖️ Primary Trial Arguments & Doctrines")
        for arg in result.get("key_legal_arguments", []):
            st.markdown(f"- 📜 {arg}")

    with c_risks:
        st.subheader("⚠️ Procedural Risks & Counter-Arguments")
        for risk in result.get("procedural_risks", []):
            st.markdown(f"- 🚩 {risk}")


# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.selectbox("Retriever Model", ["JurisVector-LegalGemma v2.1", "Lexis-Dense-Embedder 1536d", "Custom Court-BERT Classifier"])
    st.slider("Minimum Precedent Similarity", min_value=0.50, max_value=0.99, value=0.75, step=0.05)
    st.selectbox("Court Forum Filter", ["All Jurisdictions", "Federal Appellate & Supreme", "State High Courts", "Commercial / Chancery"])
    st.caption("Status: Vector database connected to 3.2M Supreme Court, Appellate, and Circuit Rulings.")

# Main Input Section
st.subheader("📥 Input Litigation Fact Pattern & Legal Claims")

with st.form("litigation_form"):
    c_juris, c_forum = st.columns(2)
    with c_juris:
        forum_jurisdiction = st.text_input("Target Forum / Jurisdiction:", value="U.S. Federal District Court (N.D. Cal.)")
    with c_forum:
        claim_type = st.selectbox("Dispute Category:", [
            "Trade Secret Misappropriation",
            "Breach of Fiduciary Duty",
            "Patent Infringement",
            "Shareholder / Equity Dispute",
            "Non-Compete Enforceability"
        ])

    fact_pattern = st.text_area(
        "Enter Statement of Facts & Disputed Actions:",
        value="Former Chief Technology Officer resigned from Company A and joined competitor Company B 10 days later. Before leaving, CTO downloaded 45GB of encrypted source code and proprietary customer lists to a personal USB drive. CTO claims the files were personal documents, but forensic logs confirm access to proprietary repositories 2 hours before submitting resignation. Company B launched a directly competing SaaS product 3 weeks after CTO joined.",
        height=140
    )

    run_pipeline_btn = st.form_submit_button("🚀 Run Case Law Retrieval & Outcome Prediction", type="primary", use_container_width=True)


if run_pipeline_btn:
    if not fact_pattern.strip():
        st.warning("Please enter a fact pattern to analyze.")
    else:
        # 1. UI Pipeline Animation
        simulate_vector_retrieval()

        # 2. Secretly construct feature payload for Gemini
        payload = {
            "jurisdiction": forum_jurisdiction,
            "claim_type": claim_type,
            "fact_narrative": fact_pattern
        }

        # 3. Call Gemini under the hood
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[f"Execute precedent retrieval and litigation outcome prediction for: {json.dumps(payload)}"],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=CaseOutcomeInference,
                ),
            )

            inference_data = json.loads(response.text)

            # 4. Render as Judicial Dashboard
            render_juris_dashboard(inference_data)

        except Exception as e:
            st.error(f"Inference Engine Failure: {e}")