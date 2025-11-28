import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
FILES = {
    "Rule 1: Metabolic State (NCOR2-)": "pathway_validation_Rule_1_NCOR2-___Mito_High_PD.csv",
    "Rule 2: Iron/Stress State (FTH1+)": "pathway_validation_Rule_2_FTH1pos___Ferritin_High_PD.csv",
    "Rule 4: Fibrotic/Senescent State (PALD1+)": "pathway_validation_Rule_4_PALD1pos_Broad_PD.csv"
}

def clean_term_name(term):
    """Removes the (GO:xxxx) ID to make labels readable"""
    return term.split(" (GO:")[0].split(" (R-HSA")[0]

def plot_pathways():
    print("Generating Clean GO Pathway Figure...")
    
    # Setup Figure: 3 Rows, 1 Column
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False) # Made slightly wider
    plt.subplots_adjust(hspace=0.5) # More space between plots
    
    # Iterate through rules and axes
    for ax, (title, csv_file) in zip(axes, FILES.items()):
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Creating placeholder.")
            ax.text(0.5, 0.5, "Data Not Found", ha='center')
            continue

        # 1. Load Data
        df = pd.read_csv(csv_file)
        
        # --- THE FIX: FILTER FOR GO TERMS ONLY ---
        # This removes the "Jensen DISEASES" rows (like Aspergillosis/Otitis)
        # and keeps only "GO_Biological_Process"
        df_go = df[df['Gene_set'].str.contains("GO_Biological_Process")].copy()
        
        # Safety check: if no GO terms found, revert to original (or print warning)
        if df_go.empty:
            print(f"  Note: No specific GO terms found for {title}. Showing all terms.")
            top_5 = df.head(5).copy()
        else:
            top_5 = df_go.head(5).copy()
        # -----------------------------------------
        
        # 2. Process Data for Plotting
        top_5 = top_5.iloc[::-1] # Invert so best is at top
        top_5['Term_Clean'] = top_5['Term'].apply(clean_term_name)
        
        # Add tiny epsilon to avoid log(0)
        top_5['log_p'] = -np.log10(top_5['Adjusted P-value'] + 1e-300)
        
        # 3. Colors
        if "Rule 1" in title: color = '#d62728' # Red
        elif "Rule 2" in title: color = '#ff7f0e' # Orange
        else: color = '#1f77b4' # Blue

        # 4. Plot
        bars = ax.barh(top_5['Term_Clean'], top_5['log_p'], color=color, alpha=0.8)
        
        # 5. Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("-log10(Adjusted P-value)", fontsize=10)
        
        # Fix x-axis width (add 15% buffer)
        max_val = top_5['log_p'].max() if not top_5.empty else 1
        ax.set_xlim(0, max_val * 1.15)
        
        # Significance line (p=0.05 is approx 1.3)
        ax.axvline(1.3, color='gray', linestyle='--', linewidth=0.8)

        # Value Labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max_val * 0.01), bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    output_file = "pathway_validation_figure_clean.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_pathways()
