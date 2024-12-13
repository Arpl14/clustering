import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_excel('Final_data_clustering.xlsx')

# Similarity calculation function
def calculate_similarity(input_company, data, embeddings, keywords):
    # Find the index of the input company
    try:
        company_index = data[data['Company Name'] == input_company].index[0]
    except IndexError:
        return None, f"Company '{input_company}' not found in the dataset."

    # Base embeddings similarity
    similarities = cosine_similarity([embeddings[company_index]], embeddings)[0]

    # Boost similarity for keyword matches
    for idx in range(len(data)):
        if idx != company_index:  # Skip self-comparison
            description1 = set(data['description from formnext'][company_index].lower().split())
            description2 = set(data['description from formnext'][idx].lower().split())
            common_keywords = description1.intersection(description2).intersection(set(keywords))
            if common_keywords:
                similarities[idx] *= 1.2  # Apply a boost factor

    # Combine similarity with additional features
    top_matches = np.argsort(similarities)[::-1][:6]  # Top 6 to exclude the input company itself

    # Filter out the input company itself from the results
    results = []
    for idx in top_matches:
        if idx != company_index:
            results.append({
                "Company Name": data.iloc[idx]['Company Name'],
                "Similarity Score": similarities[idx],
                "Description": data.iloc[idx]['description from formnext']
            })

    return results, None

# Streamlit app
st.set_page_config(layout="wide")
st.title("Top 5 Similar Companies Finder")

# Load data
data = load_data()

# Preprocess the 'description from formnext' and generate embeddings
data['description from formnext'] = data['description from formnext'].fillna('').astype(str)
embeddings = model.encode(data['description from formnext'].tolist(), show_progress_bar=False)

# Define keywords for boosting similarity
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

# User input for company name
input_company = st.text_input("Enter the Company Name:", placeholder="E.g., XYZ Ltd.")

if input_company:
    results, error = calculate_similarity(input_company, data, embeddings, keywords)

    if error:
        st.error(error)
    else:
        # Display results
        st.subheader(f"Top 5 Most Similar Companies to '{input_company}':")
        results_df = pd.DataFrame(results)
        st.table(results_df)

        # Option to download results
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(results_df)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"{input_company}_similar_companies.csv",
            mime='text/csv'
        )
