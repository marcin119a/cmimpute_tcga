import pandas as pd

path = 'data'

# Read the parquet file
df = pd.read_parquet(path + '/final_df_to_CMImpute.parquet')
print(f"Original DataFrame shape: {df.shape}")

# Set case_barcode as index (it should not be a feature column)
df = df.set_index('case_barcode')
print(f"\nDataFrame shape after setting index: {df.shape}")

# Add dummy species column (model expects species data)
# Assuming all samples are human
df['species_human'] = True

# Reorder columns: Species -> Primary sites -> CpG sites
cols = list(df.columns)
species_cols = [col for col in cols if 'species_' in col]
primary_cols = [col for col in cols if 'primary_site_' in col]
cpg_cols = [col for col in cols if 'primary_site_' not in col and 'species_' not in col]

# Create new column order
new_column_order = species_cols + primary_cols + cpg_cols
df = df[new_column_order]

print(f"DataFrame shape after reordering: {df.shape}")
print(df.head())

# Find column indices for reference
cols = list(df.columns)
primary_cols_idx = [i for i, col in enumerate(cols) if 'primary_site_' in col]
species_cols_idx = [i for i, col in enumerate(cols) if 'species_' in col]
cpg_start = primary_cols_idx[-1] + 1

print(f"\nColumn structure:")
print(f"  - Species: columns {species_cols_idx[0]} to {species_cols_idx[-1]} ({len(species_cols_idx)} columns)")
print(f"  - Primary site (tissue): columns {primary_cols_idx[0]} to {primary_cols_idx[-1]} ({len(primary_cols_idx)} columns)")
print(f"  - CpG sites (data): columns {cpg_start} to {len(cols)-1} ({len(cols)-cpg_start} columns)")

# Save as pickle
output_path = path + '/train.pickle'
df.to_pickle(output_path)
print(f"\nSaved DataFrame to {output_path}")
print(f"\nFor training command, use these indices:")
print(f"  t_start: {primary_cols[0]}")
print(f"  t_end: {primary_cols[-1]}")
print(f"  s_start: {species_cols[0]}")
print(f"  s_end: {species_cols[-1]}")
print(f"  d_start: {species_cols[-1]}")

