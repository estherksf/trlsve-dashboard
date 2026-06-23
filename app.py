import streamlit as st
import pandas as pd
import plotly.express as px
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TRL-SVE Analyst Dashboard", layout="wide", page_icon="🛡️")

# --- LOAD REAL 50K DATASET ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("pipeline_g_results.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure pipeline_g_results.csv is uploaded.")
        return pd.DataFrame()

df_results = load_data()

# --- HEADER ---
st.title("🛡️ TRL-SVE Analyst Dashboard")
st.markdown("**Real-Data Explorer:** Querying the Pipeline G validation subset.")
st.divider()

if not df_results.empty:
    # --- SIDEBAR: SEARCH REAL RECORDS ---
    st.sidebar.header("🔍 Search Database")
    
    # Let the user pick a random user ID from the dataset
    sample_users = df_results['user_id'].head(200).tolist()
    selected_user = st.sidebar.selectbox("Select User ID:", sample_users)
    
    analyze_button = st.sidebar.button("Retrieve TRL-SVE Output", type="primary")

    if analyze_button:
        with st.spinner("Retrieving Pipeline G architectural logs..."):
            time.sleep(0.5) 
            
        # Extract the exact row for this user
        record = df_results[df_results['user_id'] == selected_user].iloc[0]
        
        # Determine Status (Using 50% as the threshold)
        is_fraud = record['predicted_probability'] > 50.0 
        status = "🚨 HIGH RISK: Deceptive Review Detected" if is_fraud else "✅ LOW RISK: Genuine Review"
        status_color = "error" if is_fraud else "success"

        # --- DISPLAY REAL REVIEW DATA ---
        st.subheader("Review Metadata")
        st.markdown(f"**Target Business:** `{record['product_id']}`")
        st.info(f'"{record["review_text"]}"')

        # --- DISPLAY REAL MODEL OUTPUTS ---
        if status_color == "error": 
            st.error(f"### {status}")
        else: 
            st.success(f"### {status}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pipeline G Probability", f"{record['predicted_probability']:.1f}%")
        col2.metric("True Label (Ground Truth)", "Deceptive" if record['true_label'] in [-1, 0] else "Genuine")
        col3.metric("Action Space", "Continuous Soft-Voting")
        st.divider()

        # --- DISPLAY REAL PPO WEIGHTS ---
        st.subheader("🧠 Explainable AI: Real PPO Agent Soft-Voting Breakdown")
        
        weights = {
            "Expert": ["Relational (LRGCL)", "Behavioral (BiLSTM)", "Semantic (Longformer)"],
            "Weight": [record['weight_relational'], record['weight_behavioral'], record['weight_semantic']],
            "Features Extracted": [
                "Bipartite graph adjacency mappings",
                "30-step chronological rating sequence", # <--- UPDATED HERE
                "Sparse-attention contextual semantics"
            ]
        }
        df_weights = pd.DataFrame(weights)

        col_chart, col_text = st.columns([1, 1])

        with col_chart:
            fig = px.bar(
                df_weights, 
                x="Weight", 
                y="Expert", 
                orientation='h', 
                text="Weight", 
                color="Expert", 
                color_discrete_sequence=["#005088", "#11CAA0", "#e2e8f0"]
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, xaxis_title="Influence Weight (%)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with col_text:
            st.markdown("<br>", unsafe_allow_html=True)
            for i, row in df_weights.iterrows():
                st.markdown(f"**{row['Expert']} ({row['Weight']:.1f}%)**")
                st.caption(f"↳ *Analyzed: {row['Features Extracted']}*")
                st.markdown("---")