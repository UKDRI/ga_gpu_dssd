import pandas as pd
import os
import re
import random
import argparse  # Import argparse

# --- Configuration ---
# 1. Define our biomarker set S (from the 10-hour GA run)
BIOMARKER_GENES = ['MT-CO2', 'FAM19A5', 'HMCN2', 'CFAP46']

# 2. Define input files
X_FILE = os.path.join('sc_data', 'X_binary.csv')
Y_FILE = os.path.join('sc_data', 'y_labels.csv')

# 3. Define output directories
PD_DIR = 'emm_pd_data'
CTRL_DIR = 'emm_ctrl_data'
RESULTS_DIR = 'emm_results' 

# 4. Define output filenames
PD_DSNAME = 'pd_run'
CTRL_DSNAME = 'ctrl_run'

# --- End Configuration ---

def write_arff(dataframe, model_genes, out_path, relation_name):
    """
    Converts a Pandas DataFrame to a simple ARFF file.
    Writes description genes as INTEGER and model genes as NOMINAL {0, 1}.
    """
    print(f"Writing ARFF file to {out_path}...")
    
    # Open with newline='\r\n' to force Windows line endings for DSSD
    with open(out_path, 'w', newline='\r\n', encoding='utf-8-sig') as f:
        # --- FIX: Use lowercase @relation ---
        f.write(f"@relation {relation_name}\n\n")
        
        # Write attributes
        for col in dataframe.columns:
            # Sanitize the column name (remove quotes)
            sanitized_col = re.sub(r"['\"]", '', col)
            # --- FIX: Add quotes back to the attribute name ---
            col_name = f"'{sanitized_col}'"
            
            # --- FIX: Use lowercase @attribute ---
            if col in model_genes:
                # The "model" (biomarker) genes are NOMINAL (binary)
                # --- FIX: Remove space from {0, 1} ---
                f.write(f"@attribute {col_name} {{0,1}}\n")
            else:
                # The "description" (regulator) genes are NUMERIC (INTEGER)
                f.write(f"@attribute {col_name} INTEGER\n")
            
        # --- FIX: Use lowercase @data ---
        f.write("\n@data\n")
        
        # Write data
        for row in dataframe.itertuples(index=False):
            f.write(','.join(map(str, row)) + '\n')
    print(f"Successfully wrote {len(dataframe)} rows and {len(dataframe.columns)} columns to {out_path}")

def write_emm(description_genes, model_genes, out_path):
    """Writes the .emm file to define model/description attributes."""
    print(f"Writing EMM file to {out_path}...")
    
    # Create quoted lists of gene names
    # The .emm file *does* seem to want quotes, so we keep them here.
    quoted_model_genes = []
    for g in model_genes:
        # --- FIX: f-strings cannot contain backslashes for escaping ---
        sanitized_g = re.sub(r"['\"]", '', g)
        quoted_model_genes.append(f"'{sanitized_g}'")
        
    quoted_description_genes = []
    for g in description_genes:
        sanitized_g = re.sub(r"['\"]", '', g)
        quoted_description_genes.append(f"'{sanitized_g}'")
    
    # Open with newline='\r\n' to force Windows line endings
    with open(out_path, 'w', newline='\r\n', encoding='utf-8-sig') as f:
        f.write("# --- EMM Attribute Definition ---\n")
        # Use the correct keywords for DSSD
        f.write(f"modelAtts = {','.join(quoted_model_genes)}\n")
        f.write(f"descriptionAtts = {','.join(quoted_description_genes)}\n")
        
    print(f"Successfully wrote EMM file ({len(model_genes)} model, {len(description_genes)} description).")

def main():
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Prepare EMM data by splitting into PD/Ctrl and creating .arff/.emm files.")
    parser.add_argument('--k', type=int, default=None, 
                        help="Number of random description features to select. Default: use all (~996).")
    args = parser.parse_args()
    num_random_features = args.k
    
    print("--- Starting EMM Data Preparation ---")
    
    # Load all data
    print(f"Loading data from {X_FILE} and {Y_FILE}...")
    X_df = pd.read_csv(X_FILE)
    y_df = pd.read_csv(Y_FILE)
    
    all_genes = X_df.columns.tolist()
    
    # Ensure biomarkers exist in the dataset
    for gene in BIOMARKER_GENES:
        if gene not in all_genes:
            print(f"!!! CRITICAL ERROR: Biomarker gene '{gene}' not found in {X_FILE} columns.")
            return
            
    # --- Feature Selection Logic ---
    # 1. Identify all potential description genes (everything except biomarkers)
    all_description_genes = [g for g in all_genes if g not in BIOMARKER_GENES]
    
    # 2. Apply Random Sampling based on --k argument
    if num_random_features and num_random_features > 0 and num_random_features < len(all_description_genes):
        print(f"--- RANDOM SAMPLING ENABLED: Selecting {num_random_features} random description genes ---")
        # random.seed(42) # Uncomment to make the selection reproducible
        description_genes = random.sample(all_description_genes, num_random_features)
    else:
        if num_random_features is not None:
            print(f"--- WARNING: --k value ({num_random_features}) is invalid or > available genes ({len(all_description_genes)}). Using ALL features. ---")
        else:
             print(f"--- USING ALL FEATURES: Using all {len(all_description_genes)} description genes ---")
        description_genes = all_description_genes

    # 3. Define the final list of columns to keep
    #    Important: Description genes first, then biomarker genes
    genes_to_keep = description_genes + BIOMARKER_GENES
    print(f"Total genes in this run: {len(genes_to_keep)} ({len(description_genes)} desc, {len(BIOMARKER_GENES)} model)")

    # 4. Filter and Split Data
    X_df['disease_label'] = y_df['x']
    genes_to_keep_with_label = genes_to_keep + ['disease_label']
    
    # Use .copy() to avoid SettingWithCopyWarning
    X_df_filtered = X_df[genes_to_keep_with_label].copy()
    
    # Split data into PD (label=1) and Ctrl (label=0) and drop label
    pd_data = X_df_filtered[X_df_filtered['disease_label'] == 1].drop(columns=['disease_label'])
    ctrl_data = X_df_filtered[X_df_filtered['disease_label'] == 0].drop(columns=['disease_label'])
    
    # Reorder columns to ensure they match our genes_to_keep list exactly
    pd_data = pd_data[genes_to_keep]
    ctrl_data = ctrl_data[genes_to_keep]
    
    print(f"Data split: {len(pd_data)} PD samples, {len(ctrl_data)} Ctrl samples.")
    
    # --- Create Directories ---
    os.makedirs(PD_DIR, exist_ok=True)
    os.makedirs(CTRL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Created directories: {PD_DIR}, {CTRL_DIR}, {RESULTS_DIR}")

    # --- Write PD Files ---
    pd_arff_path = os.path.join(PD_DIR, f"{PD_DSNAME}.arff")
    pd_emm_path = os.path.join(PD_DIR, f"{PD_DSNAME}.emm")
    # Pass BIOMARKER_GENES so write_arff knows which ones are model attributes
    write_arff(pd_data, BIOMARKER_GENES, pd_arff_path, PD_DSNAME)
    write_emm(description_genes, BIOMARKER_GENES, pd_emm_path)
    
    # --- Write Ctrl Files ---
    ctrl_arff_path = os.path.join(CTRL_DIR, f"{CTRL_DSNAME}.arff")
    ctrl_emm_path = os.path.join(CTRL_DIR, f"{CTRL_DSNAME}.emm")
    write_arff(ctrl_data, BIOMARKER_GENES, ctrl_emm_path, CTRL_DSNAME)
    write_emm(description_genes, BIOMARKER_GENES, ctrl_emm_path)
    
    print("\n--- EMM Data Preparation Complete ---")
    print(f"You are now ready to write your .conf files and run the 'run_emm.slurm' script.")

if __name__ == "__main__":
    main()

