import pandas as pd
import re
import os

# Get the directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
file_311 = os.path.join(BASE_DIR, '311_Service_Requests_from_2020_to_Present_20260415.csv')
file_nypd = os.path.join(BASE_DIR, 'NYPD_Complaint_Data_Historic_20260415.csv')
output_file = os.path.join(BASE_DIR, 'preprocessed_data.csv')

def extract_precinct(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    match = re.search(r'(\d+)', val)
    if match:
        return int(match.group(1))
    return None

def extract_year_month(val):
    if pd.isna(val) or len(str(val)) < 10:
        return None
    val = str(val).strip()
    # Assuming format MM/DD/YYYY... extract YYYY-MM
    # Slices: MM is 0:2, YYYY is 6:10
    return f"{val[6:10]}-{val[0:2]}"

print("Processing 311 dataset...")
chunk_size = 1000000
precinct_311_stats = {}

for i, chunk in enumerate(pd.read_csv(file_311, chunksize=chunk_size, usecols=['Police Precinct', 'Status', 'Created Date'], low_memory=False)):
    chunk['Precinct'] = chunk['Police Precinct'].apply(extract_precinct)
    chunk['YearMonth'] = chunk['Created Date'].apply(extract_year_month)
    valid_chunks = chunk.dropna(subset=['Precinct', 'YearMonth'])
    
    for name, group in valid_chunks.groupby(['Precinct', 'YearMonth']):
        total = len(group)
        unresolved = len(group[group['Status'].str.contains('Open|Pending|Unspecified', case=False, na=False)])
        
        if name not in precinct_311_stats:
            precinct_311_stats[name] = {'total_311': 0, 'unresolved_311': 0}
        
        precinct_311_stats[name]['total_311'] += total
        precinct_311_stats[name]['unresolved_311'] += unresolved
        
    print(f"Processed 311 chunk {i+1}")

df_311 = pd.DataFrame.from_dict(precinct_311_stats, orient='index')
df_311.index.names = ['Precinct', 'YearMonth']
df_311.reset_index(inplace=True)

print("Processing NYPD dataset...")
precinct_nypd_stats = {}

for i, chunk in enumerate(pd.read_csv(file_nypd, chunksize=chunk_size, usecols=['ADDR_PCT_CD', 'CMPLNT_FR_DT'], low_memory=False)):
    chunk['Precinct'] = pd.to_numeric(chunk['ADDR_PCT_CD'], errors='coerce')
    chunk['YearMonth'] = chunk['CMPLNT_FR_DT'].apply(extract_year_month)
    valid_chunks = chunk.dropna(subset=['Precinct', 'YearMonth'])
    
    for name, group in valid_chunks.groupby(['Precinct', 'YearMonth']):
        total = len(group)
        if name not in precinct_nypd_stats:
            precinct_nypd_stats[name] = {'total_crimes': 0}
        precinct_nypd_stats[name]['total_crimes'] += total
        
    print(f"Processed NYPD chunk {i+1}")

df_nypd = pd.DataFrame.from_dict(precinct_nypd_stats, orient='index')
df_nypd.index.names = ['Precinct', 'YearMonth']
df_nypd.reset_index(inplace=True)

print("Merging datasets...")
# Merge on Precinct and YearMonth
merged_df = pd.merge(df_311, df_nypd, on=['Precinct', 'YearMonth'], how='inner')

# Calculate unresolved proportion
merged_df['unresolved_proportion'] = merged_df['unresolved_311'] / merged_df['total_311']

# Filter out bad precincts (e.g., 0)
merged_df = merged_df[merged_df['Precinct'] > 0]

# Sort to keep things organized
merged_df = merged_df.sort_values(by=['Precinct', 'YearMonth'])

merged_df.to_csv(output_file, index=False)
print(f"Saved preprocessed data to {output_file}")
print(merged_df.head())
