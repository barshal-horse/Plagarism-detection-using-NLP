import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# --- Core Plagiarism Analysis Functions ---

@st.cache_resource
def load_bert_model():
    """Loads the Sentence-BERT model and caches it to avoid reloading."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def analyze_documents(docs, method, lsi_components=100): ## MODIFIED ##: Added parameter for LSI
    """
    Single function to run the selected analysis method.
    Returns a sorted list of tuples with file pairs and their similarity score.
    """
    if not docs or len(docs) < 2:
        return []

    filenames = list(docs.keys())
    texts = list(docs.values())

    if "TF-IDF" in method:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

    ## NEW ##: Added the LSI analysis block
    elif "LSI" in method:
        # 1. Create TF-IDF vectors first
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 2. Reduce dimensionality with TruncatedSVD (LSI)
        # Ensure n_components is less than the number of documents
        actual_components = min(lsi_components, len(filenames) - 1)
        if actual_components < 1:
            st.error("LSI requires at least 2 documents to run.")
            return []
            
        svd = TruncatedSVD(n_components=actual_components)
        lsi_matrix = svd.fit_transform(tfidf_matrix)
        
        # 3. Calculate similarity on the LSI matrix
        sim_matrix = cosine_similarity(lsi_matrix)

    else:  # Sentence-BERT
        model = load_bert_model()
        embeddings = model.encode(texts, convert_to_tensor=True)
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()

    results = []
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            results.append((filenames[i], filenames[j], sim_matrix[i, j]))
    
    return sorted(results, key=lambda x: x[2], reverse=True)

# --- Streamlit User Interface ---

# Configure the page
st.set_page_config(page_title="Plagiarism Checker Dashboard", layout="wide")

# App title
st.title("📊 Plagiarism Analysis Dashboard")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_files = st.file_uploader("Upload .txt files", type="txt", accept_multiple_files=True)
    
    ## MODIFIED ##: Added LSI to the list of methods
    method = st.radio(
        "Choose Analysis Method",
        ("Sentence-BERT (Advanced)", "Latent Semantic Indexing (LSI)", "TF-IDF (Classic)")
    )
    
    ## NEW ##: Added a conditional input for LSI components
    lsi_num_components = 100
    if "LSI" in method:
        lsi_num_components = st.number_input(
            "Number of Concepts (LSI)",
            min_value=1, max_value=500, value=100,
            help="The number of underlying topics to find. A smaller number groups more words; a larger number is more specific."
        )

    threshold = st.slider("Similarity Threshold (%) for Summary", 0, 100, 75)
    run_button = st.button("Analyze Documents")

# Main panel logic
if not run_button:
    st.info("Upload your documents and click 'Analyze Documents' to generate the dashboard.")
    st.stop()

if not uploaded_files or len(uploaded_files) < 2:
    st.error("Please upload at least two .txt files to analyze.")
    st.stop()

# --- Main Analysis and Dashboard ---
documents = {file.name: file.getvalue().decode("utf-8") for file in uploaded_files}

with st.spinner("Analyzing... This may take a moment."):
    ## MODIFIED ##: Pass the LSI components parameter to the analysis function
    results = analyze_documents(documents, method, lsi_components=lsi_num_components)

st.success("Analysis complete!")

if not results:
    st.warning("Could not generate any results.")
    st.stop()

df = pd.DataFrame(results, columns=["File 1", "File 2", "Similarity Score"])
# Ensure score is properly bounded between 0 and 1
df['Similarity Score'] = df['Similarity Score'].clip(0, 1)

# --- 1. Key Metrics ---
st.header("Dashboard Summary")
total_pairs = len(df)
flagged_pairs = df[df["Similarity Score"] * 100 > threshold].shape[0]
highest_score = df["Similarity Score"].max()
average_score = df["Similarity Score"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pairs Compared", f"{total_pairs}")
col2.metric(f"Pairs Above {threshold}%", f"{flagged_pairs}")
col3.metric("Highest Similarity", f"{highest_score:.2%}")
col4.metric("Average Similarity", f"{average_score:.2%}")

st.divider()

# --- 2. Charts in Columns ---
col1, col2 = st.columns(2)

with col1:
    MAX_BARS_TO_DISPLAY = 25

    if len(df) > MAX_BARS_TO_DISPLAY:
        display_df = df.head(MAX_BARS_TO_DISPLAY).copy()
        chart_title = f"Top {MAX_BARS_TO_DISPLAY} Most Similar Pairs"
    else:
        display_df = df.copy()
        chart_title = "Similarity of All Compared Pairs"

    st.subheader(chart_title)
    
    display_df["Pair"] = display_df["File 1"] + " & " + display_df["File 2"]
    
    # Create a single, sorted DataFrame for the chart
    sorted_display_df = display_df.sort_values(by="Similarity Score", ascending=True)

    fig_pairs = px.bar(
        sorted_display_df,
        x="Similarity Score", 
        y="Pair", 
        orientation='h',
        text=sorted_display_df["Similarity Score"].apply(lambda x: f'{x:.2%}'),
        labels={'Similarity Score': 'Similarity Score (%)'}
    )
    fig_pairs.update_layout(xaxis_tickformat='.0%', yaxis_title="")
    st.plotly_chart(fig_pairs, use_container_width=True)

with col2:
    st.subheader("Distribution of Scores")
    fig_hist = px.histogram(
        df, x="Similarity Score", nbins=20,
        labels={'x': 'Similarity Score Bins', 'y': 'Number of Pairs'}
    )
    fig_hist.update_layout(xaxis_tickformat='.0%')
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# --- 3. Per-Document Analysis ---
st.subheader("Per-Document Similarity Analysis")
all_files = pd.concat([df['File 1'], df['File 2']]).unique()
doc_stats = []
for doc_name in all_files:
    doc_scores = df[(df['File 1'] == doc_name) | (df['File 2'] == doc_name)]["Similarity Score"]
    if not doc_scores.empty:
        doc_stats.append({
            "Document": doc_name,
            "Max Similarity": doc_scores.max(),
            "Average Similarity": doc_scores.mean()
        })
doc_stats_df = pd.DataFrame(doc_stats).sort_values(by="Max Similarity", ascending=False)
fig_doc_stats = px.bar(
    doc_stats_df, x="Document", y=["Max Similarity", "Average Similarity"],
    barmode='group', text_auto='.2%',
    labels={'value': 'Similarity Score', 'variable': 'Metric'}
)
fig_doc_stats.update_layout(yaxis_tickformat='.0%')
st.plotly_chart(fig_doc_stats, use_container_width=True)

st.divider()

# --- 4. Full Data Table ---
st.subheader("Full Report Data")
st.dataframe(df.style.format({"Similarity Score": "{:.2%}"}), hide_index=True, use_container_width=True)
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Report as CSV", csv, "plagiarism_report.csv", "text/csv")