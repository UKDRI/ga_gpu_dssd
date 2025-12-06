import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Loading data files...")

# Load Binary Matrix
X = pd.read_csv('sc_data/X_binary.csv') 

# Load Cell IDs
cell_ids = pd.read_csv('sc_data/cell_ids.csv')
if 'x' in cell_ids.columns:
    cell_ids = cell_ids.rename(columns={'x': 'Barcode'})
else:
    cell_ids.columns = ['Barcode']

# --- FIXED R METADATA LOADING ---
r_metadata = pd.read_csv('sc_data/pd_metadata_with_pseudotime.csv') 

# 1. Check for Index Column (Usually "Unnamed: 0")
if "Unnamed: 0" in r_metadata.columns:
    # If we ALSO have an existing 'Barcode' column, we have a collision.
    if "Barcode" in r_metadata.columns:
        print("Duplicate Barcode columns detected. Keeping the Index (Long ID)...")
        # Drop the existing metadata 'Barcode' (usually the short one)
        r_metadata = r_metadata.drop(columns=["Barcode"])
    
    # Now it is safe to rename the index to Barcode
    r_metadata = r_metadata.rename(columns={"Unnamed: 0": "Barcode"})

# 2. Safety Net: Remove any duplicate columns if they still exist
r_metadata = r_metadata.loc[:, ~r_metadata.columns.duplicated()]
# --------------------------------

# ---------------------------------------------------------
# 2. DIAGNOSTICS
# ---------------------------------------------------------
print("\n--- DIAGNOSTICS ---")
# Detect the correct time column
if 'pseudotime' in r_metadata.columns:
    time_col = 'pseudotime'
elif 'pseudotime_proxy' in r_metadata.columns:
    time_col = 'pseudotime_proxy'
else:
    candidates = [c for c in r_metadata.columns if 'time' in c.lower() or 'pc1' in c.lower()]
    if candidates:
        time_col = candidates[0]
    else:
        raise KeyError("Could not find 'pseudotime' or 'pseudotime_proxy' in R metadata!")

print(f"Using trajectory column: '{time_col}'")
print(f"Sample Python Barcode: {cell_ids['Barcode'].iloc[0]}")
print(f"Sample R Barcode:      {r_metadata['Barcode'].iloc[0]}")

# ---------------------------------------------------------
# 3. Apply Rules
# ---------------------------------------------------------
print("\nApplying rules...")
X['Barcode'] = cell_ids['Barcode']
X['Subgroup'] = 'Unassigned'

# RULE 1 (Metabolic)
mask_r1 = (
    (X['MT-ND4L'] == 1) & (X['NCOR2'] == 0) & (X['FAM19A5'] == 0) & (X['BRSK2'] == 0) & 
    (X['BIN1'] == 0) & (X['SEMA4D'] == 0) & (X['FAM102A'] == 0) & (X['ZFYVE16'] == 0) & 
    (X['ST3GAL4'] == 0) & (X['ATP6V1D'] == 0) & (X['DSTN'] == 0) & (X['CHST3'] == 0) & 
    (X['PALD1'] == 0) & (X['COX4I1'] == 0) & (X['FNTA'] == 0)
)
X.loc[mask_r1, 'Subgroup'] = 'Rule 1 (Metabolic)'

# RULE 2 (Iron/Stress)
mask_r2 = (
    (X['FTH1'] == 1) & (X['RBX1'] == 0) & (X['PALD1'] == 0) & 
    (X['SLTM'] == 0) & (X['TMEM219'] == 0)
)
X.loc[mask_r2, 'Subgroup'] = 'Rule 2 (Iron/Stress)'

# RULE 4 (Regeneration)
mask_r4 = (X['PALD1'] == 1)
X.loc[mask_r4, 'Subgroup'] = 'Rule 4 (Regeneration)'

# ---------------------------------------------------------
# 4. Merge & Filter
# ---------------------------------------------------------
print("Merging with R metadata...")
# Now that 'Barcode' is unique in both, this should work
merged_df = pd.merge(r_metadata, X[['Barcode', 'Subgroup']], on='Barcode', how='inner')
print(f"Merged Total Cells: {len(merged_df)}")

# Label Controls
if 'Disease' in merged_df.columns:
    control_labels = ['HC', 'Control', 'CTR', 'Healthy', 'HC1', 'Normal']
    is_control = merged_df['Disease'].isin(control_labels)
    merged_df.loc[is_control, 'Subgroup'] = 'Control'

plot_groups = ['Control', 'Rule 1 (Metabolic)', 'Rule 2 (Iron/Stress)', 'Rule 4 (Regeneration)']
plot_df = merged_df[merged_df['Subgroup'].isin(plot_groups)].copy()

print(f"Cells available for plotting: {len(plot_df)}")

if len(plot_df) == 0:
    print("ERROR: No cells matched. Check if R barcodes match Python barcodes exactly.")
    exit()

# ---------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------
# Auto-flip axis
control_mean = plot_df[plot_df['Subgroup']=='Control'][time_col].mean()
r4_mean = plot_df[plot_df['Subgroup']=='Rule 4 (Regeneration)'][time_col].mean()

if control_mean > r4_mean:
    print("Flipping axis so Control is on the left...")
    plot_df[time_col] = plot_df[time_col] * -1

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

my_palette = {
    "Control": "lightgrey",
    "Rule 1 (Metabolic)": "#4c72b0", 
    "Rule 2 (Iron/Stress)": "#dd8452", 
    "Rule 4 (Regeneration)": "#55a868"  
}

sns.kdeplot(
    data=plot_df,
    x=time_col,
    hue="Subgroup",
    hue_order=['Control', 'Rule 1 (Metabolic)', 'Rule 2 (Iron/Stress)', 'Rule 4 (Regeneration)'],
    fill=True,
    common_norm=False,
    palette=my_palette,
    alpha=0.5,
    linewidth=2.5
)

plt.title("Chronology of Failure: Subgroup Distribution along Disease Trajectory", fontsize=18, weight='bold', pad=20)
plt.xlabel("Pseudotime (Disease Progression)", fontsize=14)
plt.ylabel("Cell Density", fontsize=14)
plt.yticks([]) 
plt.legend(title='Subgroup', labels=['Rule 4 (Regeneration)', 'Rule 2 (Iron/Stress)', 'Rule 1 (Metabolic)', 'Control'], 
           loc='upper right', frameon=False)
sns.despine(left=True)

plt.tight_layout()
plt.savefig("PD_Pseudotime_Chronology.png", dpi=300, bbox_inches='tight')
plt.show()

print("Success! Check PD_Pseudotime_Chronology.png")
