import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# These must match the output filenames from the previous script
FILES = {
    "Rule 1: Metabolic State": "pathway_validation_Rule_1_MAST.csv",
    "Rule 2: Iron/Stress State": "pathway_validation_Rule_2_MAST.csv",
    "Rule 4: Fibrotic State": "pathway_validation_Rule_4_MAST.csv"
}

def clean_term_name(term):
    return term.split(" (GO:")[0]

def plot_pathways():
    print("Generating Figure...")
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    plt.subplots_adjust(hspace=0.5)
    
    for ax, (title, csv_file) in zip(axes, FILES.items()):
        if not os.path.exists(csv_file):
            ax.text(0.5, 0.5, "Data Not Found", ha='center')
            continue

        df = pd.read_csv(csv_file)
        # We already filtered for GO in the previous step (by selecting only that database)
        
        top_5 = df.head(5).copy()
        top_5 = top_5.iloc[::-1]
        top_5['Term_Clean'] = top_5['Term'].apply(clean_term_name)
        top_5['log_p'] = -np.log10(top_5['Adjusted P-value'] + 1e-300)
        
        if "Rule 1" in title: color = '#d62728'
        elif "Rule 2" in title: color = '#ff7f0e'
        else: color = '#1f77b4'
        
        bars = ax.barh(top_5['Term_Clean'], top_5['log_p'], color=color, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("-log10(Adj. P-value)", fontsize=10)
        ax.axvline(1.3, color='gray', linestyle='--')
        
        max_val = top_5['log_p'].max()
        ax.set_xlim(0, max_val * 1.15)
        for bar in bars:
            ax.text(bar.get_width() + (max_val*0.01), bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("pathway_validation_figure.png", dpi=300, bbox_inches='tight')
    print("Saved pathway_validation_figure.png")
    plt.show()

if __name__ == "__main__":
    plot_pathways()
