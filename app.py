import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Text Classification - IndoBERT",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "lekreys/indobert-hate-speech"
    
    try:
        with st.spinner("Loading model..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            
            device = torch.device('cpu')
            model = model.to(device)
            
        return model, tokenizer, device, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

LABELS = ['pemerintah', 'aparat', 'umum', 'non']
LABEL_NAMES = {
    'pemerintah': 'Pemerintah',
    'aparat': 'Aparat',
    'umum': 'Umum',
    'non': 'Non-Kategori'
}

def predict_text(text, model, tokenizer, device, threshold=0.5):
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    predictions = (probs >= threshold).astype(int)
    results = {
        label: {
            'predicted': bool(pred),
            'probability': float(prob)
        }
        for label, pred, prob in zip(LABELS, predictions, probs)
    }
    
    return results, probs

def create_probability_chart(probabilities):
    fig = go.Figure(data=[
        go.Bar(
            x=[LABEL_NAMES[label] for label in LABELS],
            y=probabilities,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Category",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False,
    )
    
    return fig

st.title("Indonesian Text Classification")
st.subheader("Multi-label Classification using IndoBERT Model")

model, tokenizer, device, model_loaded = load_model_and_tokenizer()

if not model_loaded:
    st.error("Model failed to load. Please check the configuration.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for classification"
    )
    
    st.divider()
    
    st.header("Model Information")
    st.info(f"""
    **Model:** IndoBERT Multilabel
    
    **Categories:** 4 labels
    
    **Max Length:** 128 tokens
    
    **Device:** {device}
    
    **Status:** Loaded
    """)
    
    st.divider()
    
    st.header("Categories")
    for label in LABELS:
        st.write(f"• {LABEL_NAMES[label]}")

tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Analysis", "Analysis History"])

with tab1:
    st.subheader("Analyze Single Text")
    
    text_input = st.text_area(
        "Enter Indonesian text to analyze:",
        height=200,
        placeholder="Type or paste Indonesian text here..."
    )
    
    analyze_btn = st.button("Analyze Text", use_container_width=True, type="primary")
    
    if analyze_btn and text_input:
        with st.spinner("Analyzing..."):
            results, probabilities = predict_text(text_input, model, tokenizer, device, threshold)
            
            detected_labels = [LABEL_NAMES[label] for label, result in results.items() if result['predicted']]
            
            st.divider()
            
            if detected_labels:
                st.success(f"**Detected Categories:** {', '.join(detected_labels)}")
            else:
                st.info("No categories detected above the threshold.")
            
            st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)
            
            st.subheader("Detailed Results")
            
            cols = st.columns(2)
            for idx, (label, result) in enumerate(results.items()):
                with cols[idx % 2]:
                    status = "Detected" if result['predicted'] else "Not Detected"
                    st.metric(
                        label=LABEL_NAMES[label],
                        value=f"{result['probability']:.1%}",
                        delta=status
                    )
            
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                'detected': len(detected_labels),
                'categories': ", ".join(detected_labels) if detected_labels else "None"
            })

with tab2:
    st.subheader("Batch Text Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file (must have 'text' column)",
        type=['csv'],
        help="CSV file should contain a column named 'text' with Indonesian text"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column")
        else:
            st.success(f"Loaded {len(df)} texts")
            
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Analyze All Texts", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_list = []
                
                for idx, text in enumerate(df['text']):
                    status_text.text(f"Processing {idx + 1}/{len(df)}...")
                    progress_bar.progress((idx + 1) / len(df))
                    
                    results, probs = predict_text(str(text), model, tokenizer, device, threshold)
                    
                    row_result = {'text': text}
                    for label, result in results.items():
                        row_result[f'{label}_predicted'] = result['predicted']
                        row_result[f'{label}_probability'] = result['probability']
                    
                    results_list.append(row_result)
                
                results_df = pd.DataFrame(results_list)
                
                st.success("Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                total_detected = results_df[[f'{label}_predicted' for label in LABELS]].any(axis=1).sum()
                
                with col1:
                    st.metric("Total Texts", len(results_df))
                with col2:
                    st.metric("Categories Detected", total_detected)
                with col3:
                    st.metric("No Category", len(results_df) - total_detected)
                
                st.subheader("Category Distribution")
                
                category_counts = {
                    LABEL_NAMES[label]: results_df[f'{label}_predicted'].sum()
                    for label in LABELS
                }
                
                fig = px.bar(
                    x=list(category_counts.keys()),
                    y=list(category_counts.values()),
                    labels={'x': 'Category', 'y': 'Count'}
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

with tab3:
    st.subheader("Analysis History")
    
    if 'history' in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Analyses", len(history_df))
        with col2:
            st.metric("Categories Detected", len(history_df[history_df['detected'] > 0]))
        
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No analysis history yet. Start analyzing texts in the Single Text tab.")

st.divider()
st.caption("Indonesian Text Classification System • Powered by IndoBERT • Built with Streamlit")