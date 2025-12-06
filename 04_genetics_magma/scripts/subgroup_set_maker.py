import pandas as pd
import glob
import os

# 1. Define your file list
files = [
    "seurat_markers_Rule1_MAST_SIGNIFICANT.csv",
    "seurat_markers_Rule2_MAST_SIGNIFICANT.csv",
    "seurat_markers_Rule4_MAST_SIGNIFICANT.csv"
]

# 2. Open the output file
with open("subgroups.set", "w") as outfile:
    for filename in files:
        # Extract the rule name from the filename (e.g., "Rule1")
        # Adjust the split logic if your naming varies
        rule_name = filename.split('_')[2] 
        
        # Read the CSV
        # index_col=0 tells pandas that the first column (Gene Names) is the index
        try:
            df = pd.read_csv(filename, index_col=0)
            
            # Get the list of genes
            genes = df.index.tolist()
            
            # 3. (OPTIONAL) MAP TO ENTREZ IDS HERE
            # If you need to convert symbols to IDs, you would do it here.
            # Example: genes = [symbol_to_id_dict.get(g, g) for g in genes]

            # 4. Format the line: "SetName Gene1 Gene2 Gene3..."
            # Using tabs or spaces as delimiters
            line = f"{rule_name}\t" + "\t".join(genes) + "\n"
            
            outfile.write(line)
            print(f"Processed {rule_name}: {len(genes)} genes found.")
            
        except FileNotFoundError:
            print(f"Warning: Could not find file {filename}")

print("Done! 'subgroups.set' created.")
