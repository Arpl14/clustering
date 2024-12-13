import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import base64
import matplotlib.pyplot as plt

# Helper function to load data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# Function to generate embeddings and process data
def process_data(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['description from formnext'].tolist(), show_progress_bar=False)
    
    # Example of processing embeddings and returning processed data
    # You will include your clustering logic here
    return embeddings  # This should be replaced with actual processed data

# Helper function to download data as Excel
def get_table_download_link(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
        writer.save()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="processed_data.xlsx">Download processed data</a>'
    return href

# Set up the title of the app
st.title('Company Clustering Analysis')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if st.button('Process Data'):
        processed_data = process_data(data)
        # Assume processed_data is a DataFrame for display; replace with your actual output
        st.write(processed_data)  # Display data or results
        st.markdown(get_table_download_link(processed_data), unsafe_allow_html=True)

        # Optional: Displaying a plot
        # Assuming you have clustering labels in 'processed_data'
        if 'Cluster' in processed_data.columns:
            fig, ax = plt.subplots()
            clusters = processed_data['Cluster'].nunique()
            colors = plt.cm.viridis(np.linspace(0, 1, clusters))
            for i, color in enumerate(colors):
                ax.scatter(processed_data[processed_data['Cluster'] == i]['Feature1'], processed_data[processed_data['Cluster'] == i]['Feature2'], color=color, label=f'Cluster {i}')
            ax.legend()
            st.pyplot(fig)

# Instructions for further steps, etc.
st.markdown("""
#### Instructions
- Upload your data file.
- Click on 'Process Data' to perform clustering.
- Download the results if needed.
""")