import glob
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# --- NEW: Import shared utils ---
# Assumes ga_utils.py is in the same directory
from ga_utils import (
    parse_rules_from_text, 
    re_encode_rule_from_string, 
    get_rule_matches_cpu
)


def plot_fitness_histogram(real_fitness, all_perm_fitnesses, output_path):
    """
    Generates a histogram of permuted fitness scores with a red line for
    the real fitness score.
    """
    if real_fitness is None or not all_perm_fitnesses:
        print("Cannot generate fitness histogram: missing real or permuted data.")
        return

    plt.figure(figsize=(10, 6))
    
    # Use 50 bins, or fewer if not many data points
    bins = min(50, len(set(all_perm_fitnesses))) 
    if bins < 2: bins = 2
    
    plt.hist(all_perm_fitnesses, bins=bins, color='gray', alpha=0.7, edgecolor='black', label='Permuted (Null) Scores')
    
    plt.axvline(real_fitness, color='red', linestyle='--', linewidth=2, label=f'Real Score: {real_fitness:.6f}')
    
    plt.title('Permutation Test Fitness Distribution', fontsize=16)
    plt.xlabel('Fitness Score (WRAcc)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Fitness histogram saved to: {output_path}")
    except Exception as e:
        print(f"Error saving fitness histogram: {e}")
    plt.close()

def plot_subgroup_analysis(task_0_rules, task_0_params, output_path):
    """
    Generates a back-to-back horizontal bar chart for the Rank 1 rules,
    plus the global distribution.
    """
    if not task_0_rules or 'num_positives' not in task_0_params or 'num_negatives' not in task_0_params:
        print("Cannot generate subgroup plot: missing Task 0 rule data or global positive/negative counts.")
        if not task_0_rules:
            print("  Reason: task_0_rules list is empty.")
        if 'num_positives' not in task_0_params:
            print("  Reason: 'num_positives' not found in task_0_params.")
        if 'num_negatives' not in task_0_params:
             print("  Reason: 'num_negatives' not found in task_0_params.")
        return

    labels = []
    pos_counts = [] # PD (Yellow)
    neg_counts = [] # CTR (Blue)

    # 1. Add Global Distribution
    labels.append('Global Distribution')
    pos_counts.append(task_0_params.get('num_positives', 0))
    neg_counts.append(task_0_params.get('num_negatives', 0))

    # 2. Add Individual Rules from Rank 1
    # --- ADDED: Filter for 0-cover rules ---
    for i, rule in enumerate(task_0_rules):
        if (rule['n_pos'] + rule['n_neg']) > 0:
            # Truncate long rule strings for the label
            rule_str_short = rule['rule_str']
            if len(rule_str_short) > 70:
                rule_str_short = rule_str_short[:70] + "..."
            
            labels.append(f"Rank 1, Rule {rule['rule_num']}: {rule_str_short}")
            pos_counts.append(rule['n_pos'])
            neg_counts.append(rule['n_neg'])

    # We plot positives as negative values to go left
    pos_counts_negative = [-p for p in pos_counts]
    
    y_pos = np.arange(len(labels))
    
    plt.figure(figsize=(12, 8 + len(labels) * 0.5)) # Make plot taller for more rules
    
    # Plot Blue (CTR / Negatives) bars going right
    plt.barh(y_pos, neg_counts, align='center', color='steelblue', label='CTR (Negatives)')
    # Plot Yellow (PD / Positives) bars going left
    plt.barh(y_pos, pos_counts_negative, align='center', color='goldenrod', label='PD (Positives)')
    
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()  # Display Global at top, then Rank 1, Rule 1, etc.
    
    plt.xlabel('Count of Instances', fontsize=12)
    plt.title('Back-to-Back Class Distribution (Top Subgroups + Global)', fontsize=16)
    plt.legend(loc='lower right')
    
    # Add a vertical line at x=0
    plt.axvline(0, color='black', linewidth=0.8)
    
    # Format x-axis to be absolute values
    current_ticks = plt.gca().get_xticks()
    plt.gca().set_xticklabels([abs(int(tick)) for tick in current_ticks])
    
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Subgroup analysis plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving subgroup plot: {e}")
    plt.close()

# --- NEW: Cover Redundancy Calculation ---
def calculate_cover_redundancy(top_k_rules_text, X_data_cpu, gene_names_list, num_features):
    """
    Calculates the Cover Redundancy (CR) metric from the paper.
    CR = (1 / |S|) * sum(|c(t) - c_hat|)
    """
    print("\nCalculating Cover Redundancy for Top-K set...")
    
    # 1. Parse rule strings from the text block
    rule_strings = parse_rules_from_text(top_k_rules_text)
    if not rule_strings:
        print("Could not parse any rules for CR calculation.")
        return None
        
    print(f"  Found {len(rule_strings)} rules in the top-k set.")
    
    # 2. Re-create the cover for each rule
    all_covers = []
    for rule_str in rule_strings:
        if rule_str == "Always true":
            continue
        
        # Re-encode string "GENE=1" to a (mask, values) tensor
        mask, values = re_encode_rule_from_string(rule_str, gene_names_list, num_features)
        
        # We only need the rule tensor part (mask + values)
        rule_tensor_cpu = torch.cat([mask, values])
        
        # Get the boolean vector [n_samples] for this rule
        matches_mask = get_rule_matches_cpu(rule_tensor_cpu, X_data_cpu, num_features)
        all_covers.append(matches_mask.int()) # Store as 0s and 1s

    if not all_covers:
        print("No valid rule covers generated.")
        return None

    # 3. Calculate CR
    # Stack all cover vectors [k, n_samples]
    cover_matrix = torch.stack(all_covers)
    
    # c(t): cover count for each tuple (sample) [n_samples]
    c_t = torch.sum(cover_matrix, dim=0).float()
    
    # c_hat: expected (average) cover count
    c_hat = torch.mean(c_t)
    
    # |S|: total number of samples
    S_size = float(X_data_cpu.shape[0])
    
    # CR = (1 / |S|) * sum(|c(t) - c_hat|)
    cr_metric = torch.sum(torch.abs(c_t - c_hat)) / S_size
    
    cr_val = cr_metric.item()
    print(f"  Cover Redundancy (CR) = {cr_val:.6f}")
    return cr_val


def gather_and_analyze(results_directory):
    """
    Reads all ga_results_*.txt files, parses new format,
    calculates p-value, and generates plots.
    """
    all_perm_fitnesses = []
    real_fitness = None
    task_0_params = {}
    task_0_rules = [] # For Rank 1 rules
    
    top_rules_text = "" # To store the raw text for the summary file
    top_k_rules_text_for_cr = [] # Just the rule lines for CR
    
    task0_file_found = False

    file_pattern = os.path.join(results_directory, "ga_results_*.txt")
    result_files = glob.glob(file_pattern)

    if not result_files:
        print(f"Error: No 'ga_results_*.txt' files found in directory: {results_directory}")
        return

    print(f"Found {len(result_files)} result files in {results_directory}. Processing...")

    for f_name in result_files:
        is_task0_file = os.path.basename(f_name) == "ga_results_0.txt"
        if is_task0_file:
            task0_file_found = True

        try:
            with open(f_name, 'r') as f:
                in_params_section = False
                in_fitness_section = False
                in_rules_section = False
                
                current_rules_text = []
                current_rank = 0
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # --- State Machine for Parsing ---
                    if "--- PARAMETERS ---" in line:
                        in_params_section = True
                        in_fitness_section = False
                        in_rules_section = False
                        continue
                    elif "Best Fitness per Permutation" in line:
                        in_params_section = False
                        in_fitness_section = True
                        in_rules_section = False
                        continue
                    # --- FIX: Stop at Rank 2 for CR ---
                    elif "--- Rank 2 (Fitness:" in line:
                        in_rules_section = False # Stop parsing for CR
                    elif "Top 5 Rule Sets" in line or "Top " in line and " Rule Sets" in line: # More general
                        in_params_section = False
                        in_fitness_section = False
                        in_rules_section = True
                        if is_task0_file:
                            current_rules_text.append("\n" + line)
                            top_k_rules_text_for_cr.append(line) # Add header
                        continue
                    elif "--- Permuted Run Task ---" in line:
                        # This footer ends all sections
                        in_params_section = False
                        in_fitness_section = False
                        in_rules_section = False
                        continue
                        
                    # --- Process Lines Based on State ---
                    if in_params_section:
                        if is_task0_file: # Only need params from task 0
                            try:
                                key, val = line.split(':', 1)
                                key = key.strip()
                                val = val.strip()
                                # Try to convert to number
                                if '.' in val:
                                    task_0_params[key] = float(val)
                                else:
                                    task_0_params[key] = int(val)
                            except ValueError:
                                task_0_params[key] = val # Store as string
                            except Exception:
                                pass # Ignore malformed lines

                    elif in_fitness_section:
                        try:
                            # Skip header
                            if "Permutation_Index" in line:
                                continue
                            parts = line.split(',')
                            if len(parts) >= 3:
                                fitness = float(parts[1])
                                is_real = (parts[2].strip() == "True")

                                if is_real:
                                    if real_fitness is None:
                                        real_fitness = fitness
                                    # else: (Ignore duplicates, first one wins)
                                else:
                                    all_perm_fitnesses.append(fitness)
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse fitness line in {f_name}: {line}")

                    elif in_rules_section and is_task0_file:
                        # Add raw line to summary text
                        current_rules_text.append(line) 
                        
                        # --- FIX: Only add to CR if Rank 1 ---
                        if current_rank <= 1:
                            top_k_rules_text_for_cr.append(line) # Add all rule/stat lines

                        # --- Grab Global Stats if we don't have them ---
                        if 'num_positives' not in task_0_params:
                            pos_match = re.search(r"- Positives: \d+ / (\d+) total positives", line)
                            if pos_match:
                                task_0_params['num_positives'] = int(pos_match.group(1))
                        
                        if 'num_negatives' not in task_0_params:
                            neg_match = re.search(r"- Negatives: \d+ / (\d+) total negatives", line)
                            if neg_match:
                                task_0_params['num_negatives'] = int(neg_match.group(1))
                        
                        # --- Parse for Plot 2 ---
                        rank_match = re.search(r"--- Rank (\d+) \(Fitness: .*\) ---", line)
                        if rank_match:
                            current_rank = int(rank_match.group(1))
                        
                        # Only parse rules for Rank 1
                        if current_rank == 1:
                            # FIX: Added '^\s*' to handle leading whitespace
                            rule_match = re.search(r"^\s*- Rule (\d+): (.*)", line)
                            if rule_match:
                                rule_num = int(rule_match.group(1))
                                rule_str = rule_match.group(2)
                                # The stats are on the *next* line
                                # We'll parse them when we see the *next* rule or end of section
                                task_0_rules.append({
                                    "rule_num": rule_num,
                                    "rule_str": rule_str,
                                    "n_pos": 0, # Will be filled by next line
                                    "n_neg": 0  # Will be filled by next line
                                })
                            
                            # FIX: Added '^\s*' to handle leading whitespace
                            stat_match = re.search(r"^\s*\(Covers: \d+ total \| (\d+) Pos \| (\d+) Neg\)", line)
                            if stat_match and task_0_rules:
                                # This stat line belongs to the *last* rule added
                                n_pos = int(stat_match.group(1))
                                n_neg = int(stat_match.group(2))
                                task_0_rules[-1]['n_pos'] = n_pos
                                task_0_rules[-1]['n_neg'] = n_neg
            
            # End of file processing
            if is_task0_file and current_rules_text:
                top_rules_text = "\n".join(current_rules_text)
                
        except Exception as e:
            print(f"Error reading file {f_name}: {e}")
            continue

    # --- Analysis ---
    summary_lines = [] # Initialize here
    
    # --- NEW: Also try to parse overlap ratio ---
    overlap_ratio = None
    if task0_file_found:
        try:
            with open(os.path.join(results_directory, "ga_results_0.txt"), 'r') as f:
                for line in f:
                    or_match = re.search(r"- Overlap Ratio \(E\):\s+([\d\.]+)", line)
                    if or_match:
                        overlap_ratio = float(or_match.group(1))
                        break # Found it
        except Exception as e:
            print(f"Warning: Could not parse Overlap Ratio from ga_results_0.txt. {e}")


    # --- NEW: Load data for Cover Redundancy ---
    X_data_cpu = None
    gene_names_list = []
    
    # Need base_dir. Assume we are in 'gpu_ga' and results are in 'results_dir'
    # This finds the 'gpu_ga' directory, assuming script is run from there.
    base_dir = "." 
    data_dir_relative = "sc_data"
    x_file = os.path.join(base_dir, data_dir_relative, "X_binary.csv")
    
    if 'num_features' not in task_0_params:
        print("Warning: 'num_features' not in params. CR calc may fail.")
        
    try:
        X_df = pd.read_csv(x_file)
        gene_names_list = X_df.columns.tolist()
        # --- CHANGED: Use int8 for memory ---
        X_data_cpu = torch.tensor(X_df.values, dtype=torch.int8)
        print(f"\nLoaded {x_file} for Cover Redundancy calculation.")
        
        # --- Calculate Cover Redundancy ---
        cr_value = calculate_cover_redundancy(
            top_k_rules_text_for_cr, # Use the full text block
            X_data_cpu,
            gene_names_list,
            task_0_params.get('num_features', 1000) # Default to 1000 if not found
        )
        if cr_value is not None:
            summary_lines.append(f"\nCover Redundancy (CR) of Rank 1 Set: {cr_value:.6f}")
            if overlap_ratio is not None:
                 summary_lines.append(f"GA-Calculated Overlap Ratio (E): {overlap_ratio:.6f}")
            
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file at {x_file}.")
        print("  Skipping Cover Redundancy calculation.")
    except Exception as e:
        print(f"\nError during Cover Redundancy calculation: {e}")


    # --- Analysis ---
    print("\n--- Final Permutation Test ---")

    if real_fitness is None:
        print("Error: Real data fitness (is_real=True) not found in any result file.")
        if task0_file_found:
             print("       ga_results_0.txt was found, but the real fitness line might be missing or malformed.")
        else:
             print("       ga_results_0.txt was not found.")
        return

    print(f"Real Data Fitness: {real_fitness:.8f}")
    print(f"Total Permuted Runs Found: {len(all_perm_fitnesses)}")

    summary_lines.insert(0, f"--- PARAMETERS (from Task 0) ---")
    param_lines = [f"{key}: {val}" for key, val in task_0_params.items()]
    summary_lines.insert(1, "\n".join(param_lines)) # Insert params after header

    summary_lines.extend([
        "--- Final Permutation Test ---",
        f"Real Data Fitness: {real_fitness:.8f}",
        f"Total Permuted Runs Found: {len(all_perm_fitnesses)}"
    ])

    if len(all_perm_fitnesses) > 0:
        perm_array = np.array(all_perm_fitnesses)
        max_perm_fitness = np.max(perm_array)
        avg_perm_fitness = np.mean(perm_array)
        std_perm_fitness = np.std(perm_array)

        # Calculate p-value: (count >= real) / total_permuted
        count_better_or_equal = np.sum(perm_array >= real_fitness)
        
        # Empirical p-value: (count_better + 1) / (total_runs + 1)
        p_value_B = count_better_or_equal
        p_value_N = len(all_perm_fitnesses)
        p_value = (p_value_B + 1) / (p_value_N + 1)

        print(f"Max Permuted Fitness: {max_perm_fitness:.8f}")
        print(f"Avg Permuted Fitness: {avg_perm_fitness:.8f}")
        print(f"Std Dev Permuted Fitness: {std_perm_fitness:.8f}")
        print(f"\nEmpirical p-value (B+1)/(N+1): ({p_value_B} + 1) / ({p_value_N} + 1) = {p_value:.6f}")
        if p_value_B == 0:
            print(f"  (p-value < {1/(p_value_N+1):.6f})")


        summary_lines.extend([
            f"Max Permuted Fitness: {max_perm_fitness:.8f}",
            f"Avg Permuted Fitness: {avg_perm_fitness:.8f}",
            f"Std Dev Permuted Fitness: {std_perm_fitness:.8f}",
            f"\nEmpirical p-value (B+1)/(N+1): ({p_value_B} + 1) / ({p_value_N} + 1) = {p_value:.6f}",
            f"  (Count of permuted scores >= real score: {p_value_B})"
        ])

    else:
        print("Error: Could not find any permuted run results.")
        summary_lines.append("Error: Could not find any permuted run results.")

    # --- Append Rules to Summary ---
    if top_rules_text:
        summary_lines.append(top_rules_text)
    elif task0_file_found:
        summary_lines.append("\nTop rules section not found or empty in ga_results_0.txt.")
    else:
        summary_lines.append("\nga_results_0.txt not found, cannot display top rules.")

    # --- Save Summary ---
    summary_filename = os.path.join(results_directory, "final_summary.txt")
    try:
        with open(summary_filename, "w") as f_sum:
            f_sum.write("\n".join(summary_lines))
        print(f"\nSummary saved to: {summary_filename}")
    except Exception as e:
        print(f"\nError saving summary file: {e}")

    # --- Generate Plots ---
    plot_fitness_histogram(
        real_fitness, 
        all_perm_fitnesses, 
        os.path.join(results_directory, "fitness_histogram.png")
    )
    
    plot_subgroup_analysis(
        task_0_rules, 
        task_0_params, 
        os.path.join(results_directory, "subgroup_analysis.png")
    )

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        if not os.path.isdir(target_dir):
             print(f"Error: Provided path is not a valid directory: {target_dir}")
             sys.exit(1)
    else:
        # Default to current directory if no arg given
        target_dir = "." 
        print(f"No directory provided. Defaulting to current directory: {os.path.abspath(target_dir)}")

    print(f"--- Running Analysis on Directory: {os.path.abspath(target_dir)} ---")
    gather_and_analyze(target_dir)

