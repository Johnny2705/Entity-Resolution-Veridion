import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

def clean_text(text):
    """
    Clean and normalize the dataset text
    """
    if pd.isnull(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def create_composite_key(row):
    """
    Create a composite key for each row based on important columns
    We are creating the key by cleaning and concatenating values from the important columns
    The key acts as a unique identifier for clustering for each company
    """
    parts = []
    for col in important_columns:
        parts.append(clean_text(row[col]))
    return " ".join(parts)

# The list of important columns to be used for composite key generation
important_columns = [
    "company_name",
    "company_legal_names",
    "company_commercial_names",
    "year_founded",
    "short_description",
    "naics_2022_primary_label",
    "generated_business_tags"
]

# Reading the dataset
input_file = "veridion_entity_resolution_challenge.snappy.parquet"
df = pd.read_parquet(input_file)

# Checking if the important columns exist in the DataFrame, if not, create them with empty strings
for col in important_columns:
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found. Using an empty string as default.")
        df[col] = ""
    else:
        df[col] = df[col].fillna("")

#Generating the composite key for each record by applying the function row-wise
df['composite_key'] = df.apply(create_composite_key, axis=1)

# Setting the initial company_group to -1 (indicating no group)
df['company_group'] = -1

# Vectorization using TF-IDF to convert the text into numerical features
# We use character n-grams to capture the similarity in company names
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(df['composite_key'])

# Identifying non-zero rows as we want to cluster only those with actual data
row_sums = np.array(X.sum(axis=1)).flatten()
nonzero_mask = row_sums > 0

if nonzero_mask.sum() > 0:
    X_cluster = X[nonzero_mask].toarray()
    # Applying DBSCAN clustering, eps being the distance between points in the feature space
    # and min_samples being the minimum number of points to form a dense region
    # We use cosine distance as we are dealing with text data
    dbscan = DBSCAN(eps=0.2, min_samples=3, metric='cosine')
    cluster_labels = dbscan.fit_predict(X_cluster)
    # Mapping the cluster labels back to the original DataFrame
    df.loc[nonzero_mask, 'company_group'] = cluster_labels

# Counting the number of unique groups and ignoring the noise (labeled as -1)
unique_groups = df.loc[df['company_group'] != -1, 'company_group'].nunique()
print(f"Number of valid groups: {unique_groups}")

# Saving the updated DataFrame to CSV and Excel(only if possible for Excel as we have a limit of rows)
output_csv = "veridion_entity_resolution_composite_updated.csv"
df.to_csv(output_csv, index=False)
print(f"Updated dataset saved as CSV: {output_csv}")

if len(df) < 1048576:
    output_excel = "veridion_entity_resolution_composite_updated.xlsx"
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"Updated dataset saved as Excel: {output_excel}")
else:
    print("Number of rows exceeds Excel's limit; saved only as CSV.")

# Generating the output for duplicates and uniques

# Sorting the DataFrame by company_group for better readability
df_duplicates = df[df['company_group'] != -1].copy()
df_duplicates_sorted = df_duplicates.sort_values(by='company_group')
duplicates_excel_file = "duplicates.xlsx"
df_duplicates_sorted.to_excel(duplicates_excel_file, index=False, engine="openpyxl")
print(f"Duplicate companies saved in: {duplicates_excel_file}")

# Unique companies
df_unique = df[df['company_group'] == -1].copy()
unique_excel_file = "uniques.xlsx"
df_unique.to_excel(unique_excel_file, index=False, engine="openpyxl")
print(f"Unique companies saved in: {unique_excel_file}")
