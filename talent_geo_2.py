import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# ==========================================
# 0. 环境与配置
# ==========================================
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Lenovo Talent Brand GEO", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    .pos-card {background-color: #f0fdf4; padding: 12px; border-radius: 6px; border-left: 4px solid #22c55e; margin-bottom: 8px; font-size: 0.9em;}
    .neg-card {background-color: #fef2f2; padding: 12px; border-radius: 6px; border-left: 4px solid #ef4444; margin-bottom: 8px; font-size: 0.9em;}
    .tag-bubble {display: inline-block; background-color: #e5e7eb; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; margin: 2px 4px 10px 0px;}
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Lenovo Talent Brand: GEO Analytics Dashboard")
st.markdown("Compare ChatGPT vs. Gemini: Share of Voice, Lenovo-centric Sentiment, and Live Responses.")

# ==========================================
# 1. 核心数据获取引擎 
# ==========================================
def fetch_comparative_geo_analytics(cluster_name, specific_prompt):
    openai_client = OpenAI(api_key=OPENAI_KEY)
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    with st.spinner("1/2: Fetching live raw responses from ChatGPT and Gemini..."):
        gpt_raw = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": specific_prompt}]
        ).choices[0].message.content

        gemini_raw = gemini_model.generate_content(specific_prompt).text

    with st.spinner("2/2: AI Judge is compiling comparative GEO metrics..."):
        # 🔥 修改点 1：让 Prompt 吐出结构化的行动指南
        system_prompt = """
        You are an Employer Branding GEO Analyst. 
        I will provide a 'Keyword Cluster Persona' and the actual RAW RESPONSES from ChatGPT and Gemini.
        Return ONLY a valid JSON object matching this exact schema:
        {
          "chatgpt": {
            "share_of_voice": [{"brand": "String", "percentage": Int}],
            "lenovo_sentiment": {
              "positives": [{"topic": "String", "description": "String"}],
              "negatives": [{"topic": "String", "description": "String"}]
            },
            "cited_domains": [{"domain": "String", "citations_pct": Float, "brand_mentions_pct": Float}],
            "culture_tags": ["String", "String", "String"]
          },
          "gemini": {
            "share_of_voice": [{"brand": "String", "percentage": Int}],
            "lenovo_sentiment": {
              "positives": [{"topic": "String", "description": "String"}],
              "negatives": [{"topic": "String", "description": "String"}]
            },
            "cited_domains": [{"domain": "String", "citations_pct": Float, "brand_mentions_pct": Float}],
            "culture_tags": ["String", "String", "String"]
          },
          "action_plan": {
            "root_cause_analysis": "String (Why Lenovo performed well or poorly in these responses)",
            "quick_wins": ["String", "String (Technical SEO/GEO tweaks e.g., JSON-LD, Robots.txt, Reddit AMA)"],
            "content_strategy": "String (What long-term PR/HR content needs to be published)"
          },
          "prompt_metrics": [
            {"prompt": "String", "topic": "String", "region": "US", "visibility": "String", "sentiment": "String"}
          ]
        }
        Ensure Lenovo is included in share_of_voice. Base everything STRICTLY on the provided RAW RESPONSES.
        """

        user_content = f"Persona Cluster: {cluster_name}\nTest Prompt: {specific_prompt}\n\n=== ChatGPT Raw ===\n{gpt_raw}\n\n=== Gemini Raw ===\n{gemini_raw}"

        eval_res = openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        ).choices[0].message.content

        return gpt_raw, gemini_raw, json.loads(eval_res)

# ==========================================
# 2. UI: Keyword Cluster Selection
# ==========================================
st.header("1. Keyword Cluster Selection")
clusters = {
    "Group A (Early Career / Internships)": "Best tech internships in Raleigh-Durham for corporate culture; Top companies for new grad software engineers 2026.",
    "Group B (Senior Tech / Infrastructure)": "Top Fortune 500 hardware companies for infrastructure engineering career growth; Best WLB for senior cloud architects.",
    "Group C (Diversity & Inclusion Focus)": "Which global tech companies have the best DEI programs and female leadership representation?"
}

selected_cluster = st.selectbox("Select Target Persona:", list(clusters.keys()))
current_prompt = clusters[selected_cluster]
st.info(f"**Live Tracking Prompts:** {current_prompt}")

if st.button("🚀 Generate Comparative GEO Dashboard", type="primary"):
    if not OPENAI_KEY or not GEMINI_KEY:
        st.error("Missing API Keys. Please check your .env file.")
    else:
        gpt_raw, gemini_raw, data = fetch_comparative_geo_analytics(selected_cluster, current_prompt)
        st.divider()

        # ==========================================
        # 3. UI: Share of Voice
        # ==========================================
        st.header("2. Share of Voice: ChatGPT vs. Gemini")
        df_gpt = pd.DataFrame(data["chatgpt"]["share_of_voice"])
        df_gpt["AI Engine"] = "ChatGPT"
        df_gem = pd.DataFrame(data["gemini"]["share_of_voice"])
        df_gem["AI Engine"] = "Gemini"
        df_sov = pd.concat([df_gpt, df_gem])
        
        fig_sov = px.bar(
            df_sov, x="brand", y="percentage", color="AI Engine", barmode="group",
            text="percentage", color_discrete_sequence=["#10a37f", "#4285f4"]
        )
        fig_sov.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_sov.update_layout(yaxis_title="Mention Percentage (%)", xaxis_title="", height=400)
        st.plotly_chart(fig_sov, use_container_width=True)

        st.divider()

        # ==========================================
        # 4. UI: Lenovo Sentiment
        # ==========================================
        st.header("3. Competitive Sentiment & Citation Sources")
        col_c, col_g = st.columns(2)
        
        with col_c:
            st.subheader("🤖 ChatGPT Analysis")
            tags_html = "".join([f"<div class='tag-bubble'>🏷️ {tag}</div>" for tag in data["chatgpt"]["culture_tags"]])
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("**Lenovo Strengths (vs. Others):**")
            for pos in data["chatgpt"]["lenovo_sentiment"]["positives"]:
                st.markdown(f"<div class='pos-card'><b>{pos['topic']}:</b> {pos['description']}</div>", unsafe_allow_html=True)
            st.markdown("**Lenovo Weaknesses (vs. Others):**")
            for neg in data["chatgpt"]["lenovo_sentiment"]["negatives"]:
                st.markdown(f"<div class='neg-card'><b>{neg['topic']}:</b> {neg['description']}</div>", unsafe_allow_html=True)

        with col_g:
            st.subheader("♊ Gemini Analysis")
            tags_html = "".join([f"<div class='tag-bubble'>🏷️ {tag}</div>" for tag in data["gemini"]["culture_tags"]])
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("**Lenovo Strengths (vs. Others):**")
            for pos in data["gemini"]["lenovo_sentiment"]["positives"]:
                st.markdown(f"<div class='pos-card'><b>{pos['topic']}:</b> {pos['description']}</div>", unsafe_allow_html=True)
            st.markdown("**Lenovo Weaknesses (vs. Others):**")
            for neg in data["gemini"]["lenovo_sentiment"]["negatives"]:
                st.markdown(f"<div class='neg-card'><b>{neg['topic']}:</b> {neg['description']}</div>", unsafe_allow_html=True)

        st.divider()

        # ==========================================
        # 5. UI: 🔥 全新强化的 Action Plan
        # ==========================================
        st.header("4. 🎯 Strategic GEO Action Plan")
        plan = data["action_plan"]
        
        st.error(f"**🔍 Root Cause Analysis:**\n\n{plan['root_cause_analysis']}")
        
        st.markdown("**⚡ Quick Wins:**")
        for win in plan['quick_wins']:
            st.markdown(f"- {win}")
            
        st.success(f"**📚 Content Strategy:**\n\n{plan['content_strategy']}")

        st.divider()

        # ==========================================
        # 6. UI: View Raw AI Responses
        # ==========================================
        st.header("5. Raw Engine Output Logs")
        col_raw_c, col_raw_g = st.columns(2)
        with col_raw_c:
            with st.expander("📄 View Raw AI Response (ChatGPT)"):
                st.write(gpt_raw)
        with col_raw_g:
            with st.expander("📄 View Raw AI Response (Gemini)"):
                st.write(gemini_raw)