import pandas as pd
import os

# --- Configuration ---
# 1. Define our biomarker set S (from the 10-hour GA run)
# We'll use the genes from the simplest, most powerful rules.
BIOMARKER_GENES = ['MT-CO2', 'FAM19A5', 'HMCN2', 'CFAP46']

# 2. Define input files (relative to where you run this script)
X_FILE = os.path.join('sc_data', 'X_binary.csv')
Y_FILE = os.path.join('sc_data', 'y_labels.csv')

# 3. Define output directories
PD_DIR = 'emm_pd_data'
CTRL_DIR = 'emm_ctrl_data'
RESULTS_DIR = 'emm_results' # DSSD will write its output here

# 4. Define output filenames (DSSD requires <dsName>.arff and <dsName>.emm)
PD_DSNAME = 'pd_run'
CTRL_DSNAME = 'ctrl_run'

# --- End Configuration ---

def write_arff(dataframe, out_path, relation_name):
    """Converts a Pandas DataFrame to a simple ARFF file."""
    print(f"Writing ARFF file to {out_path}...")
    
    # --- FIX: Open with newline='\r\n' to force Windows line endings ---
    with open(out_path, 'w', newline='\r\n') as f:
        f.write(f"@RELATION {relation_name}\n\n")
        
        # Write attributes (all are binary)
        for col in dataframe.columns:
            col_name = f"'{col}'"
            
            # --- FIX: Change attribute type from {0, 1} to INTEGER ---
            # The DSSD parser seems to prefer a numeric type.
            f.write(f"@ATTRIBUTE {col_name} INTEGER\n")
            
        f.write("\n@DATA\n")
        
        # Write data
        for row in dataframe.itertuples(index=False):
            f.write(','.join(map(str, row)) + '\n')
    print(f"Successfully wrote {len(dataframe)} rows to {out_path}")

def write_emm(all_genes, model_genes, out_path):
    """Writes the .emm file to define model/description attributes."""
    print(f"Writing EMM file to {out_path}...")
    
    # DSSD is particular about quoting
    # --- FIX: Always quote all gene names ---
    quoted_model_genes = [f"'{g}'" for g in model_genes]
    
    # --- FIX: Open with newline='\r\n' to force Windows line endings ---
    with open(out_path, 'w', newline='\r\n') as f:
        f.write("# --- EMM Attribute Definition ---\n")
        f.write(f"model = {', '.join(quoted_model_genes)}\n")
        f.write("description = *\n")
    print(f"Successfully wrote EMM file.")

def main():
    print("--- Starting EMM Data Preparation ---")
    
    # Load all data
    print(f"Loading data from {X_FILE} and {Y_FILE}...")
    X_df = pd.read_csv(X_FILE)
    y_df = pd.read_csv(Y_FILE)
    
    all_genes = X_df.columns.tolist()
    
    # Ensure all biomarkers are in the data
    for gene in BIOMARKER_GENES:
        if gene not in all_genes:
            print(f"!!! CRITICAL ERROR: Biomarker gene '{gene}' not found in {X_FILE} columns.")
            return

    # Add labels to data for filtering
    X_df['disease_label'] = y_df['x']
    
    # Split data into PD (label=1) and Ctrl (label=0)
    pd_data = X_df[X_df['disease_label'] == 1].drop(columns=['disease_label'])
    ctrl_data = X_df[X_df['disease_label'] == 0].drop(columns=['disease_label'])
    
    print(f"Data split: {len(pd_data)} PD samples, {len(ctrl_data)} Ctrl samples.")
    
    # --- Create Directories ---
    os.makedirs(PD_DIR, exist_ok=True)
    os.makedirs(CTRL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Created directories: {PD_DIR}, {CTRL_DIR}, {RESULTS_DIR}")

    # --- Write PD Files ---
    pd_arff_path = os.path.join(PD_DIR, f"{PD_DSNAME}.arff")
    pd_emm_path = os.path.join(PD_DIR, f"{PD_DSNAME}.emm")
    write_arff(pd_data, pd_arff_path, PD_DSNAME)
    write_emm(all_genes, BIOMARKER_GENES, pd_emm_path)
    
    # --- Write Ctrl Files ---
    ctrl_arff_path = os.path.join(CTRL_DIR, f"{CTRL_DSNAME}.arff")
    ctrl_emm_path = os.path.join(CTRL_DIR, f"{CTRL_DSNAME}.emm")
    write_arff(ctrl_data, ctrl_arff_path, CTRL_DSNAME)
    write_emm(all_genes, BIOMARKER_GENES, ctrl_emm_path)
    
    print("\n--- EMM Data Preparation Complete ---")
    print("You are now ready to write your .conf files and run the 'run_emm.slurm' script.")

if __name__ == "__main__":
    main()

