import glob
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

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
    if not task_0_rules or not task_0_params:
        print("Cannot generate subgroup plot: missing Task 0 rule data or params.")
        return

    labels = []
    pos_counts = [] # PD (Yellow)
    neg_counts = [] # CTR (Blue)

    # 1. Add Global Distribution
    labels.append('Global Distribution')
    pos_counts.append(task_0_params.get('num_positives', 0))
    neg_counts.append(task_0_params.get('num_negatives', 0))

    # 2. Add Individual Rules from Rank 1
    for i, rule in enumerate(task_0_rules):
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
                    elif "Top 5 Rule Sets" in line:
                        in_params_section = False
                        in_fitness_section = False
                        in_rules_section = True
                        if is_task0_file:
                            current_rules_text.append("\n" + line)
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
                        
                        # --- Parse for Plot 2 ---
                        rank_match = re.search(r"--- Rank (\d+) \(Fitness: .*\) ---", line)
                        if rank_match:
                            current_rank = int(rank_match.group(1))
                        
                        # Only parse rules for Rank 1
                        if current_rank == 1:
                            rule_match = re.search(r"- Rule (\d+): (.*)", line)
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
                            
                            stat_match = re.search(r"\(Covers: \d+ total \| (\d+) Pos \| (\d+) Neg\)", line)
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

    summary_lines = [
        "--- Final Permutation Test ---",
        f"Real Data Fitness: {real_fitness:.8f}",
        f"Total Permuted Runs Found: {len(all_perm_fitnesses)}"
    ]

    if len(all_perm_fitnesses) > 0:
        perm_array = np.array(all_perm_fitnesses)
        max_perm_fitness = np.max(perm_array)
        avg_perm_fitness = np.mean(perm_array)
        std_perm_fitness = np.std(perm_array)

        # Calculate p-value: (count >= real) / total_permuted
        count_better_or_equal = np.sum(perm_array >= real_fitness)
        
        # Empirical p-value: (count_better + 1) / (total_runs + 1)
        # We assume the 'real' run is one of the runs, but we're comparing
        # it to a separate set of permuted runs.
        # Let's stick to the (B+1)/(N+1) formula.
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
```

### How to Run:

1.  Save the code above as `gather_results.py` in your main `gpu_ga` directory.
2.  Run it from your `gpu_ga` directory, passing the **results directory** as an argument:

    ```bash
    # Example for your 10-minute test run:
    python3 gather_results.py /scratch/dn-neav1/ga_test_10min
    
    # Example for your production run:
    python3 gather_results.py /scratch/dn-neav1/ga_prod_10hr
    
