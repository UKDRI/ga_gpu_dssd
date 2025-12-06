import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
FILES = {
    "Metabolic\n(Rule 1)": "Rule1_FULL.rnk",
    "Iron-Toxic\n(Rule 2)": "Rule2_FULL.rnk",
    "Fibrotic\n(Rule 4)": "Rule4_FULL.rnk"
}

GENE_SETS = 'GO_Biological_Process_2023'

def run_gsea():
    print("Running GSEA on Full Ranked Lists...")
    all_data = []

    for label, rnk_file in FILES.items():
        if not os.path.exists(rnk_file):
            print(f"Skipping {rnk_file}")
            continue
            
        print(f"  Processing {label.split()[0]}...")
        
        # Run GSEA Prerank
        # min_size=10 keeps specific terms, max_size=500 removes generic ones
        pre_res = gp.prerank(rnk=rnk_file, gene_sets=GENE_SETS,
                             threads=4, min_size=10, max_size=500, seed=42)
        
        results = pre_res.res2d.copy()
        
        # Filter: Significant (FDR < 0.05) AND Positive Enrichment (NES > 0)
        # We only want what is UP in the subgroup
        sig = results[(results['FDR q-val'] < 0.05) & (results['NES'] > 0)].copy()
        
        if sig.empty:
            print("    No significant pathways.")
            continue
            
        # Take Top 5 by NES (Normalized Enrichment Score)
        top_5 = sig.sort_values('NES', ascending=False).head(5).copy()
        top_5['Group'] = label
        
        # Clean Terms
        top_5['Term_Clean'] = top_5['Term'].apply(lambda x: x.split(" (GO:")[0])
        all_data.append(top_5)

    if not all_data: return

    # --- PLOTTING ---
    df_plot = pd.concat(all_data)
    
    # Create a Dot Plot
    plt.figure(figsize=(9, 10))
    
    # Sort for display order
    # We map groups to specific colors
    colors = {"Metabolic\n(Rule 1)": "#d62728", 
              "Iron-Toxic\n(Rule 2)": "#ff7f0e", 
              "Fibrotic\n(Rule 4)": "#1f77b4"}
    
    # Create a unified y-axis
    y_labels = []
    y_pos = []
    current_y = 0
    
    for group in FILES.keys(): # iterate in order
        group_data = df_plot[df_plot['Group'] == group]
        if group_data.empty: continue
        
        # Sort this group by NES
        group_data = group_data.sort_values('NES', ascending=True)
        
        for _, row in group_data.iterrows():
            plt.scatter(row['NES'], current_y, s=100, color=colors[group], zorder=3)
            plt.hlines(current_y, 0, row['NES'], color=colors[group], alpha=0.5, linewidth=2)
            y_labels.append(row['Term_Clean'])
            y_pos.append(current_y)
            current_y += 1
        
        current_y += 1 # Add spacer between groups
        
    plt.yticks(y_pos, y_labels, fontsize=11)
    plt.xlabel("Normalized Enrichment Score (NES)", fontsize=12)
    plt.title("GSEA: Top Enriched Pathways (Data-Driven)", fontsize=14, fontweight='bold')
    
    # Add Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=g.split('\n')[0],
                          markerfacecolor=c, markersize=10) for g, c in colors.items()]
    plt.legend(handles=handles, loc='lower right')
    
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("gsea_validation_figure.png", dpi=300, bbox_inches='tight')
    print("Saved gsea_validation_figure.png")
    plt.show()

if __name__ == "__main__":
    run_gsea()
