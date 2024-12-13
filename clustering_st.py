import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from io import BytesIO
import base64

# Helper function to load data
@st.cache_data
def load_data():
    # Load the data from the specified file
    return pd.read_excel('Final_data_clustering.xlsx')

# Helper function to create a download link for the DataFrame
def get_download_link(df):
    towrite = BytesIO()
    df.to_excel(towrite, index=False, engine='xlsxwriter')  # Write to BytesIO buffer
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()  # Convert bytes to base64
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Clustered_Data.xlsx">Download Excel file</a>'
    return href

# Enhanced similarity function
def enhanced_similarity(index1, index2, embeddings, additional_features, data, keywords):
    # Check for direct keyword match in keywords column
    keywords1 = set(str(data['Keywords and extra information'][index1]).split(', '))
    keywords2 = set(str(data['Keywords and extra information'][index2]).split(', '))
    if keywords1.intersection(keywords2):
        return 1.0  # Return maximum similarity score if direct keyword match

    # Base description similarity with keyword boosting
    description_similarity = cosine_similarity([embeddings[index1]], [embeddings[index2]])[0][0]
    text1_words = set(data['description from formnext'][index1].lower().split())
    text2_words = set(data['description from formnext'][index2].lower().split())
    common_keywords = text1_words.intersection(text2_words).intersection(set(keywords))
    if common_keywords:
        description_similarity *= 1.2  # Apply boost factor if common keywords are found

    # Weights for features
    description_weight = 0.5
    am_process_material_weight = 0.15
    country_category_weight = 0.10

    # Additional features similarity
    additional_similarity = cosine_similarity([additional_features[index1]], [additional_features[index2]])[0][0]

    # Weighted sum of similarities
    total_similarity = (description_weight * description_similarity +
                        am_process_material_weight * additional_similarity +
                        country_category_weight * additional_similarity)

    return total_similarity

# Streamlit page configuration
st.title('Company Clustering Analysis')

# Load data
data = load_data()

# Preprocess the 'Keywords and extra information' column to handle missing or non-string values
data['Keywords and extra information'] = data['Keywords and extra information'].fillna('').astype(str)

# Display data table
st.write("Data Loaded and Displayed Below:")
st.write(data.head())

if st.button('Process and Cluster Data'):
    # Generate embeddings for descriptions
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['description from formnext'].tolist(), show_progress_bar=False)

    # One-hot encode additional categorical columns
    categorical_columns = ['Type fo AM process', 'Type of Material', 'Country', 'Category']
    encoder = OneHotEncoder()
    encoded_columns = encoder.fit_transform(data[categorical_columns]).toarray()
    encoded_columns /= encoded_columns.max(axis=0)  # Normalize the encoded columns

    # Keywords list assumed predefined
    keywords = [
        'Titanium parts', 'laser', 'dlp', 'powder', 'software', 'metal', 'SLM', 'Large-scale production',
        'Cold spray technology', 'Robotic repair', 'Low-pressure cold spray', 'Glass', 'DLP', 'LED-based DLP',
        'Multilaser', 'lithography', 'net-shape metal', 'Castequivalent', '2PP', 'High-resolution laser lithography',
        'Microfabrication', 'Ceramic 3D printing', 'LCM', 'Metal powder bed fusion', 'SEBM',
        'Projection micro stereolithography', 'High-temperature materials', 'PEEK', 'PEKK', 'ULTEM',
        'Thermoplastic', 'Thermoplastic 3D printing', 'Binder jetting technology', 'Selective laser sintering',
        'Laser cladding', 'Photopolymers', 'Flexible material 3D printing', 'High-performance technical ceramics',
        'Digital metal additive manufacturing', 'Powder metallurgy', 'High-value alloys', 'Nickel', 'Cobalt',
        'Superalloys', 'Material recycling', 'Fused filament fabrication', 'filament', 'Industrial 3D printing',
        'Custom medical devices', 'Dentistry', 'Audiology', 'Stereolithography', 'Precision engineering',
        'Aerospace components', 'Defence applications', 'Energy sector', 'Sustainable manufacturing', 'Green technology',
        'Rapid prototyping', 'Laser sintering', 'Electronics manufacturing', 'Biomedical applications', 'Automotive industry',
        'Jewelry production', 'Education sector', 'Research and development', 'Optical components', 'Building and construction',
        'Custom alloys', 'High-strength materials', 'Composite materials', 'High-performance polymers', 'Metal alloys',
        'Polymeric microparts', 'CAD/CAM solutions', 'Microstructure analysis', 'Layered manufacturing', 'Complex geometries',
        'Innovative materials', 'Advanced ceramics'
    ]

    # Calculate the enhanced similarity matrix
    n = data.shape[0]
    enhanced_similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            enhanced_similarity_matrix[i][j] = enhanced_similarity(i, j, embeddings, encoded_columns, data, keywords)

    # Clustering
    distance_matrix = 1 - enhanced_similarity_matrix
    clustering = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='complete')
    clusters = clustering.fit_predict(distance_matrix)
    data['Cluster'] = clusters

    # Display clustered data
    st.write("Clustered Data:")
    st.write(data)
    st.markdown(get_download_link(data), unsafe_allow_html=True)
