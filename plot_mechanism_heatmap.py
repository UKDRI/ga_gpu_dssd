import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# 1. The Files (Update these if your filenames differ slightly)
#FILES = {
#    "Rule 1\n(Metabolic)": "enrichment_Rule_1_NCOR2-___Mito_High_PD.csv",
#    "Rule 2\n(Iron/Stress)": "enrichment_Rule_2_FTH1pos___Ferritin_High_PD.csv",
#    "Rule 4\n(Fibrotic)": "enrichment_Rule_4_PALD1pos_Broad_PD.csv"
#}
##Updated files
FILES = {
    "Rule 1\n(Metabolic)": "enrichment_Rule_1_Metabolic___STRICT_PD.csv",
    "Rule 2\n(Iron/Stress)": "enrichment_Rule_2_Iron___STRICT_PD.csv",
    "Rule 4\n(Fibrotic)": "enrichment_Rule_4_Fibrotic___STRICT_PD.csv"
}

# 2. The Genes to Highlight (The "Story")
# We manually select the best markers for each state to make the plot clean.
GENES_OF_INTEREST = [
    # --- Mito Markers (Rule 1) ---
    "MT-ND4", "MT-CO3", "MT-ATP6", "MT-ND1",
    # --- Iron/Stress Markers (Rule 2) ---
    "FTH1", "FTL", "CRYAB", "PLP1",
    # --- Fibrosis/Regen Markers (Rule 4) ---
    "LINGO1", "COL27A1", "COL5A1", "RBFOX3", "GSE1"
]

def plot_heatmap():
    print("Generating Mechanism Heatmap...")
    
    # Initialize matrix: Rows=Genes, Cols=Rules
    data_matrix = pd.DataFrame(index=GENES_OF_INTEREST, columns=FILES.keys())
    data_matrix = data_matrix.fillna(0.0) # Default to 0 (no enrichment)

    # Fill Data
    for rule_label, csv_file in FILES.items():
        if not os.path.exists(csv_file):
            print(f"Warning: File {csv_file} not found. Skipping.")
            continue
            
        df = pd.read_csv(csv_file)
        
        # Create a dictionary for fast lookup {Gene: Difference}
        # We use 'Difference' (e.g., +0.25 means 25% more frequent)
        enrichment_map = dict(zip(df['Gene'], df['Difference']))
        
        for gene in GENES_OF_INTEREST:
            if gene in enrichment_map:
                data_matrix.loc[gene, rule_label] = enrichment_map[gene]
            else:
                # If gene not in the "Enriched" CSV, it means diff was small.
                # We leave it as 0.0 (Grey)
                pass

    # --- Plotting ---
    plt.figure(figsize=(6, 8))
    
    # Create Heatmap
    # cmap="RdBu_r": Red=High, Blue=Low, White=Zero
    # center=0 ensures 0 is white
    ax = sns.heatmap(data_matrix, annot=True, fmt=".2f", 
                     cmap="RdBu_r", center=0, 
                     linewidths=.5, cbar_kws={'label': 'Enrichment Difference (Subgroup - Background)'})
    
    plt.title("Differential Enrichment of Key Pathway Markers", fontsize=14)
    plt.ylabel("Gene", fontsize=12)
    plt.xlabel("Subgroup Rule", fontsize=12)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    output_file = "mechanism_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_heatmap()
