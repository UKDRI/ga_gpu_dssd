import pandas as pd
import os

# 1. Load Data
input_path = "./sc_data/X_binary.csv"
df = pd.read_csv(input_path)
print(f"Original shape: {df.shape}")

# 2. Comprehensive Banned List (Derived from your text)
# We exclude the genes from Rule 3 because that rule was statistical noise (1 cell)
banned_genes = [
    # --- Rule 4 (The Broad Driver) ---
    "PALD1",
    
    # --- Rule 2 (The Iron/Ferritin Subgroup) ---
    "FTH1", "RBX1", "SLTM", "TMEM219", 
    
    # --- Rule 1 (The Complex Metabolic Subgroup) ---
    "MT-ND4L", # The only positive anchor in Rule 1
    "NCOR2", "FAM19A5", "BRSK2", "BIN1", "SEMA4D", 
    "FAM102A", "ZFYVE16", "ST3GAL4", "ATP6V1D", 
    "DSTN", "CHST3", "COX4I1", "FNTA"
]

# 3. Check what's actually in the file (to avoid errors if names differ slightly)
existing_ban = [g for g in banned_genes if g in df.columns]
missing_ban = [g for g in banned_genes if g not in df.columns]

if missing_ban:
    print(f"Warning: These genes were not found in columns: {missing_ban}")

# 4. Drop them
df_blinded = df.drop(columns=existing_ban)
print(f"Dropped {len(existing_ban)} genes.")
print(f"New shape: {df_blinded.shape}")

# 5. Save blinded dataset
output_dir = "./sc_data_blinded"
os.makedirs(output_dir, exist_ok=True)

df_blinded.to_csv(os.path.join(output_dir, "X_binary.csv"), index=False)

# Copy labels so the GA can find them
os.system(f"cp ./sc_data/y_labels.csv {output_dir}/y_labels.csv")

print(f"Ready for Shadow Run! Data saved to: {output_dir}")
