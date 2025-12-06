import pandas as pd

# 1. Load the file
filename = "seurat_markers_Rule4_MAST_TOP100.csv"
df = pd.read_csv(filename)

# 2. Fix the Column Name
# The first column (index) often has no name in R exports. 
# We rename the first column to "Gene"
df.rename(columns={df.columns[0]: "Gene"}, inplace=True)

# 3. Filter and Sort
# We want Upregulated genes (log2FC > 0) sorted by significance (lowest p-value)
df_up = df[df['avg_log2FC'] > 0]
top_50_genes = df_up.sort_values(by="p_val_adj", ascending=True).head(50)['Gene'].tolist()

# 4. Print for Copy-Pasting into STRING
print("--- Top 50 Genes for STRING ---")
print('\n'.join(top_50_genes))
