import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import gseapy as gp

# --- CONFIGURATION: Use Seurat Results ---
# These contain p-values derived from the broader Seurat dataset using mast

FILES = {
    "Rule 1: Metabolic State (NCOR2-)": "seurat_markers_Rule1_MAST_SIGNIFICANT.csv",
    "Rule 2: Iron/Stress State (FTH1+)": "seurat_markers_Rule2_MAST_SIGNIFICANT.csv",
    "Rule 4: Fibrotic/Senescent State (PALD1+)": "seurat_markers_Rule4_MAST_SIGNIFICANT.csv"
}

GENE_SETS = ['GO_Biological_Process_2023']

def run_validation_and_plot():
    print("Generating GO Validation from Seurat Results...")
    
    # Store top terms for plotting
    plot_data = {}

    for title, csv_file in FILES.items():
        if not os.path.exists(csv_file):
            print(f"Skipping {title} (File not found: {csv_file})")
            continue
            
        # 1. Load Seurat Markers
        # Seurat saves gene names in the first column (often unnamed)
        df = pd.read_csv(csv_file)
        # Rename first column to 'Gene' if it's unnamed
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "Gene"}, inplace=True)
        
        # 2. Get the Gene List
        # These are already filtered for significance by your R script
        gene_list = df['Gene'].tolist()
        print(f"  {title}: Found {len(gene_list)} significant genes.")
        
        if len(gene_list) < 5:
            print("    Not enough genes for enrichment.")
            continue

        # 3. Run Enrichr
        # We use the standard 'Human' background because Seurat tested against the transcriptome
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=GENE_SETS,
                         organism='Human',
                         outdir=None)
        
        results = enr.results
        # Filter for significance
        sig_results = results[results['Adjusted P-value'] < 0.05].sort_values("Adjusted P-value")
        
        if sig_results.empty:
            print("    No significant pathways found.")
            continue
            
        # Save top 5 for plotting
        plot_data[title] = sig_results.head(5).copy()
        print(f"    Found {len(sig_results)} pathways.")

    # --- PLOTTING ---
    if not plot_data:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    plt.subplots_adjust(hspace=0.5)
    
    rules_ordered = list(FILES.keys())
    
    for i, rule_name in enumerate(rules_ordered):
        ax = axes[i]
        if rule_name not in plot_data:
            ax.text(0.5, 0.5, "No Significant Pathways", ha='center')
            continue
            
        df_plot = plot_data[rule_name]
        
        # Clean Names
        df_plot['Term_Clean'] = df_plot['Term'].apply(lambda x: x.split(" (GO:")[0])
        df_plot['log_p'] = -np.log10(df_plot['Adjusted P-value'] + 1e-300)
        df_plot = df_plot.iloc[::-1] # Reverse for bar chart
        
        # Colors
        if "Rule 1" in rule_name: color = '#d62728'
        elif "Rule 2" in rule_name: color = '#ff7f0e'
        else: color = '#1f77b4'
        
        # Bar Chart
        bars = ax.barh(df_plot['Term_Clean'], df_plot['log_p'], color=color, alpha=0.8)
        
        ax.set_title(rule_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("-log10(Adjusted P-value)", fontsize=10)
        ax.axvline(1.3, color='gray', linestyle='--', linewidth=0.8)
        
        # Labels
        max_val = df_plot['log_p'].max()
        ax.set_xlim(0, max_val * 1.15)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max_val * 0.01), bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("pathway_validation_figure_seurat.png", dpi=300, bbox_inches='tight')
    print("Saved to pathway_validation_figure_seurat.png")
    plt.show()

if __name__ == "__main__":
    run_validation_and_plot()
