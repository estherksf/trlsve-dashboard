import streamlit as st
import pandas as pd
import plotly.express as px
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TRL-SVE Dashboard", layout="wide", page_icon="🛡️")

# --- LOAD REAL DATASET (ZIP COMPRESSION) ---
@st.cache_data
def load_data():
    try:
        # Pandas automatically handles the extraction of the .zip file
        df = pd.read_csv("pipeline_g_results.zip")
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure pipeline_g_results.csv.zip is uploaded to GitHub.")
        return pd.DataFrame()

df_results = load_data()

# --- HEADER ---
st.title("🛡️ TRL-SVE Dashboard")
st.markdown("**Real-Data Explorer:** Querying the Pipeline G validation subset.")
st.divider()

if not df_results.empty:
    # --- SIDEBAR: SEARCH REAL RECORDS ---
    st.sidebar.header("🔍 Search Database")
    
    # Toggle for Search Mode
    search_mode = st.sidebar.radio(
        "Select Search Mode:", 
        ["Direct User Lookup", "Explore via Filters"]
    )
    
    st.sidebar.divider()
    
    if search_mode == "Direct User Lookup":
        st.sidebar.markdown("**Targeted Investigation:**")
        all_users = sorted(df_results['user_id'].astype(str).unique().tolist())
        selected_user = st.sidebar.selectbox(
            "Search User ID:", 
            all_users,
            help="Click the box and start typing to instantly find a user."
        )
        analyze_button = st.sidebar.button("Retrieve TRL-SVE Output", type="primary")

    else:
        st.sidebar.markdown("**Pattern Discovery:**")
        # 1. Filter by True Label
        label_filter = st.sidebar.selectbox(
            "1. Filter by True Label:",
            ["All Reviews", "Deceptive Only (1)", "Genuine Only (0)"]
        )
        
        # Apply Label Filter
        filtered_df = df_results.copy()
        if label_filter == "Deceptive Only (1)":
            filtered_df = filtered_df[filtered_df['true_label'] == 1]
        elif label_filter == "Genuine Only (0)":
            filtered_df = filtered_df[filtered_df['true_label'] == 0]
            
        # 2. Filter by Product ID
        product_list = ["All Products"] + sorted(filtered_df['product_id'].astype(str).unique().tolist())
        product_filter = st.sidebar.selectbox("2. Filter by Target Business:", product_list)
        
        if product_filter != "All Products":
            filtered_df = filtered_df[filtered_df['product_id'] == product_filter]
            
        # 3. Select User ID from filtered list
        available_users = filtered_df['user_id'].astype(str).tolist()
        
        if len(available_users) == 0:
            st.sidebar.warning("No records match these exact filters.")
            selected_user = None
            analyze_button = st.sidebar.button("Retrieve TRL-SVE Output", type="primary", disabled=True)
        else:
            selected_user = st.sidebar.selectbox(
                f"3. Select User ID ({len(available_users)} available):", 
                available_users,
                help="Click the box and start typing to instantly find a user."
            )
            analyze_button = st.sidebar.button("Retrieve TRL-SVE Output", type="primary")

    # --- EXECUTE ANALYSIS ---
    if analyze_button and selected_user is not None:
        with st.spinner("Retrieving Pipeline G architectural logs..."):
            time.sleep(0.5) 
            
        # Extract the exact row for this user
        record = df_results[df_results['user_id'].astype(str) == str(selected_user)].iloc[0]
        
        # Determine Status (Using 50% as the threshold)
        is_fraud = record['predicted_probability'] >= 50.0 
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

        # --- DYNAMIC EVALUATION LOGIC ---
        # 1 = Deceptive (Fake), 0 = Genuine
        pred_class = 1 if record['predicted_probability'] >= 50.0 else 0
        true_class = int(record['true_label'])
        
        # Calculate the exact Confusion Matrix outcome
        if pred_class == 1 and true_class == 1:
            eval_result = "✅ True Positive"
        elif pred_class == 0 and true_class == 0:
            eval_result = "✅ True Negative"
        elif pred_class == 1 and true_class == 0:
            eval_result = "❌ False Positive"
        elif pred_class == 0 and true_class == 1:
            eval_result = "❌ False Negative"

        col1, col2, col3 = st.columns(3)
        col1.metric("Pipeline G Probability", f"{record['predicted_probability']:.1f}%")
        col2.metric("True Label (Ground Truth)", "Deceptive" if true_class == 1 else "Genuine")
        col3.metric("Model Evaluation", eval_result)
        st.divider()

        # --- DISPLAY REAL PPO WEIGHTS ---
        st.subheader("🧠 Explainable AI: Real PPO Agent Soft-Voting Breakdown")
        
        weights = {
            "Expert": ["Relational (LRGCL)", "Behavioral (BiLSTM)", "Semantic (Longformer)"],
            "Weight": [record['weight_relational'], record['weight_behavioral'], record['weight_semantic']],
            "Features Extracted": [
                "Bipartite graph adjacency mappings",
                "30-step chronological rating sequence",
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
