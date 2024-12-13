import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_excel('Final_data_clustering.xlsx')

# Function to calculate similarity
def calculate_similarity(input_company, data, embeddings, encoded_columns, keywords):
    try:
        company_index = data[data['Company'] == input_company].index[0]
    except IndexError:
        return None, f"Company '{input_company}' not found in the dataset."

    # Initialize similarities
    similarities = cosine_similarity([embeddings[company_index]], embeddings)[0]

    for idx in range(len(data)):
        if idx != company_index:  # Skip self-comparison
            # Description-based similarity boosting
            description1 = set(data['description from formnext'][company_index].lower().split())
            description2 = set(data['description from formnext'][idx].lower().split())
            common_keywords = description1.intersection(description2).intersection(set(keywords))
            if common_keywords:
                similarities[idx] *= 1.2  # Apply boost factor for description keywords
            
            # Boost similarity for exact match in 'Keywords and extra information'
            keywords1 = set(data['Keywords and extra information'][company_index].split(', '))
            keywords2 = set(data['Keywords and extra information'][idx].split(', '))
            if keywords1.intersection(keywords2):
                similarities[idx] += 0.3  # Add a boost for exact matches
        
            # Add weightage for additional columns
            additional_similarity = cosine_similarity([encoded_columns[company_index]], [encoded_columns[idx]])[0][0]
            similarities[idx] += 0.5 * additional_similarity  # Adjust weightage as required

    # Get top 5 matches excluding the input company
    top_matches = np.argsort(similarities)[::-1][:6]
    results = []
    for idx in top_matches:
        if idx != company_index:
            results.append({
                "Company Name": data.iloc[idx]['Company'],
                "Similarity Score": similarities[idx],
                "Description": data.iloc[idx]['description from formnext'],
                "Type of AM Process": data.iloc[idx]['Type fo AM process'],
                "Type of Material": data.iloc[idx]['Type of Material'],
                "Category": data.iloc[idx]['Category'],
                "Country": data.iloc[idx]['Country']
            })

    return company_index, results, None

# Streamlit app
st.set_page_config(layout="wide")
st.title("Top 5 Similar Companies Finder with Detailed Descriptions")

# Load data
data = load_data()

# Preprocess and encode categorical columns
data['description from formnext'] = data['description from formnext'].fillna('').astype(str)
data['Keywords and extra information'] = data['Keywords and extra information'].fillna('').astype(str)

# Generate embeddings for descriptions
embeddings = model.encode(data['description from formnext'].tolist(), show_progress_bar=False)

# One-hot encode additional categorical columns
categorical_columns = ['Type fo AM process', 'Type of Material', 'Country', 'Category']
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns]).toarray()
encoded_columns /= encoded_columns.max(axis=0)  # Normalize the encoded columns

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
    company_index, results, error = calculate_similarity(input_company, data, embeddings, encoded_columns, keywords)

    if error:
        st.error(error)
    else:
        # Display input company details
        st.subheader(f"Details for Input Company: '{input_company}'")
        input_company_details = data.iloc[company_index]
        st.write(input_company_details.to_frame().T)

        # Display results
        st.subheader(f"Top 5 Most Similar Companies to '{input_company}':")
        results_df = pd.DataFrame(results)
        st.write(results_df)

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
