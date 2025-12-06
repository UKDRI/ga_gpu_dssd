import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. Configuration: Map Rules to their Specific Files
# ---------------------------------------------------------
# We map each subgroup to the specific CSVs where its key biology hides.
# Based on your file list:
file_map = {
    "R1:Metabolic": [
        "pathway_validation_Rule_1_MAST_bg_disease_genetics.csv", # Contains Mitochondrial/Leigh hits
        "pathway_validation_Rule_1_MAST_bg_go_pathway.csv"       # Contains Oxidoreductase hits
    ],
    "R2:Iron/Stress": [
        "pathway_validation_Rule_2_MAST_bg_disease_genetics.csv", # Contains Iron HPO terms
        "pathway_validation_Rule_2_MAST_bg_go_pathway.csv"       # Contains Granule/Vesicle terms
    ],
    "R4:Regeneration": [
        "pathway_validation_Rule_4_MAST_bg_regulators.csv",       # Contains EGR1, KLF4, TFAP2A
        "pathway_validation_Rule_4_MAST_bg_go_pathway.csv",       # Contains Axon Guidance, Metal Ion Transport
        "pathway_validation_Rule_4_MAST_bg_disease_genetics.csv"  # Contains Joint Laxity (ECM)
    ]
}

# ---------------------------------------------------------
# 2. Define the "Story Terms" to Filter For
# ---------------------------------------------------------
# Since the files contain hundreds of terms, we tell the script which specific
# keywords or exact terms to look for to tell our story.
story_terms = {
    "R1:Metabolic": [
        "Mitochondrial myopathy", "Leber hereditary optic neuropathy", 
        "Leigh disease", "Lactic acidosis", "Oxidoreductase"
    ],
    "R2:Iron/Stress": [
        "Iron homeostasis", "Transition element", "Ficolin-1", "Granule"
    ],
    "R4:Regeneration": [
        "EGR1", "KLF4", "Axon guidance", "Joint laxity", "Metal Ion Transport"
    ]
}

# ---------------------------------------------------------
# 3. Data Loading & Filtering Engine
# ---------------------------------------------------------
plot_data = []

print("Processing files...")

for group, file_list in file_map.items():
    for filename in file_list:
        if os.path.exists(filename):
            try:
                # Load file
                df_temp = pd.read_csv(filename)
                
                # Normalize column names (just in case)
                df_temp.columns = [c.strip() for c in df_temp.columns]
                
                # Look for our story terms
                # We use a loose string match so "Abnormality of iron..." catches "Iron"
                keywords = story_terms[group]
                
                for term in keywords:
                    # Find rows where the Term column contains our keyword (case insensitive)
                    # Assumes column is named 'Term' and p-value is 'Adjusted P-value'
                    match = df_temp[df_temp['Term'].str.contains(term, case=False, na=False)]
                    
                    if not match.empty:
                        # Take the most significant hit for this keyword
                        best_hit = match.sort_values(by="Adjusted P-value").iloc[0]
                        
                        plot_data.append({
                            "Group": group,
                            "Term": best_hit["Term"], # Use the full name from the file
                            "Adj P-value": best_hit["Adjusted P-value"],
                            "Keyword Category": term # Helper to know which keyword triggered this
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: File not found - {filename}")

# Create DataFrame
df = pd.DataFrame(plot_data)

# Drop duplicates if a term was found multiple times
df = df.drop_duplicates(subset=["Group", "Term"])


# ---------------------------------------------------------
# INTERMEDIATE STEP: Clean up the names for the Poster
# ---------------------------------------------------------
# We create a dictionary to rename the messy long terms to clean short ones
name_cleaner = {
    "Intramolecular Oxidoreductase Activity, Transposing C=C Bonds (GO:0016863)": "Oxidoreductase Activity",
    "EGR1 20690147 ChIP-Seq ERYTHROLEUKEMIA Human": "EGR1 Targets",
    "Abnormality of iron homeostasis (HP:0011031)": "Iron Homeostasis Defect",
    "Abnormality of transition element cation homeostasis (HP:0011030)": "Cation Homeostasis Defect",
    "Ficolin-1-Rich Granule Lumen (GO:1904813)": "Ficolin-1 Rich Granules",
    "Metal Ion Transport (GO:0030001)": "Metal Ion Transport"
}

# Apply the renaming. If the term isn't in the dict, keep the original name.
df["Term"] = df["Term"].replace(name_cleaner)
# ---------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------
if not df.empty:
    # Calculate Significance Size
    df['-log10(P-value)'] = -np.log10(df['Adj P-value'])
    
    # Setup Plot
    plt.figure(figsize=(9, 8))
    sns.set_style("whitegrid")
    
    # Plot
    scatter = sns.scatterplot(
        data=df,
        x="Group",
        y="Term",
        size="-log10(P-value)",
        hue="Group",
        sizes=(200, 1000),
        palette="deep",
        alpha=0.9,
        edgecolor="black",
        linewidth=1
    )
    
    # Formatting
    plt.title("Distinct Functional Failure Modes in PD", fontsize=18, weight='bold', pad=20)
    plt.xlabel("", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=11)
    plt.margins(x=0.2, y=0.1)
    # Legend


    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        borderaxespad=0., 
        title="Significance (-log10 P)",
        labelspacing=2.5,  # INCREASE THIS: Spreads the dots out vertically
        borderpad=1.5,     # INCREASE THIS: Adds padding inside the box so dots don't touch the line
        frameon=True       # Optional: Set to False if you want to remove the box border entirely
    )

    plt.subplots_adjust(right=0.4)
    plt.tight_layout()
    plt.savefig("PD_Enrichr_DotPlot_From_Files.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot generated successfully!")
else:
    print("No matching data found. Check if files are in the same folder as the script.")
