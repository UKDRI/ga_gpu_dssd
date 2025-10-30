import glob
import os
import sys
import numpy as np # For easier statistics




## How to Run

#1.  Make sure `gather_results.py` is saved in your main `gpu_ga` directory.
#2.  Run it from that directory (since that's where the `ga_results_*.txt` files ended up):
#   ```bash
#  python3 gather_results.py


def gather_and_analyze(results_directory):
    """
    Reads ga_results_*.txt files, calculates permutation p-value,
    and prints/saves a summary.
    """
    all_perm_fitnesses = []
    real_fitness = None
    top_rules_text = ""
    task0_file_found = False

    # Define the pattern to search for result files
    file_pattern = os.path.join(results_directory, "ga_results_*.txt")
    result_files = glob.glob(file_pattern)

    if not result_files:
        print(f"Error: No 'ga_results_*.txt' files found in directory: {results_directory}")
        return

    print(f"Found {len(result_files)} result files in {results_directory}. Processing...")

    # Loop through all result files found
    for f_name in result_files:
        is_task0_file = os.path.basename(f_name) == "ga_results_0.txt"
        if is_task0_file:
            task0_file_found = True

        try:
            with open(f_name, 'r') as f:
                in_fitness_section = False
                in_rules_section = False
                current_rules = []

                for line in f:
                    line = line.strip()
                    if not line: # Skip blank lines
                        continue

                    # --- Parsing Logic ---
                    if "Best Fitness per Permutation" in line:
                        in_fitness_section = True
                        in_rules_section = False
                        continue
                    if "Top Rule Sets" in line:
                        in_fitness_section = False
                        in_rules_section = True
                        current_rules.append(line) # Keep the header
                        continue
                    if "Permuted Run Task:" in line:
                         in_fitness_section = False
                         in_rules_section = False
                         continue # End parsing for this file

                    # --- Process Lines ---
                    if in_fitness_section:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 3:
                                fitness = float(parts[1])
                                is_real = (parts[2].strip() == "True")

                                if is_real:
                                    if real_fitness is not None:
                                        print(f"Warning: Found multiple 'real' fitness values. Using the first one from {f_name}.")
                                    else:
                                        real_fitness = fitness
                                else:
                                    all_perm_fitnesses.append(fitness)
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse fitness line in {f_name}: {line}")
                            continue

                    elif in_rules_section and is_task0_file:
                        # Only store rules from task 0 file
                        current_rules.append(line)

                # Store rules from task 0 file
                if is_task0_file and current_rules:
                    top_rules_text = "\n".join(current_rules)

        except Exception as e:
            print(f"Error reading file {f_name}: {e}")
            continue # Skip to next file if one fails

    # --- Analysis ---
    print("\n--- Final Permutation Test ---")

    if real_fitness is None:
        print("Error: Real data fitness (is_real=True) not found in any result file.")
        if task0_file_found:
             print("       ga_results_0.txt was found, but the real fitness line might be missing or malformed.")
        else:
             print("       ga_results_0.txt was not found.")
        return

    print(f"Real Data Fitness: {real_fitness:.6f}")
    print(f"Total Permuted Runs Found: {len(all_perm_fitnesses)}")

    summary_lines = [
        "--- Final Permutation Test ---",
        f"Real Data Fitness: {real_fitness:.6f}",
        f"Total Permuted Runs Found: {len(all_perm_fitnesses)}"
    ]

    if len(all_perm_fitnesses) > 0:
        perm_array = np.array(all_perm_fitnesses)
        max_perm_fitness = np.max(perm_array)
        avg_perm_fitness = np.mean(perm_array)
        std_perm_fitness = np.std(perm_array)

        # Calculate p-value: (count >= real) / total_permuted
        # Add 1 to numerator and denominator for empirical p-value robustness if desired
        # count_better_or_equal = np.sum(perm_array >= real_fitness)
        # p_value = (count_better_or_equal + 1) / (len(all_perm_fitnesses) + 1)
        # Or simpler version:
        count_better_or_equal = np.sum(perm_array >= real_fitness)
        p_value = count_better_or_equal / len(all_perm_fitnesses) if len(all_perm_fitnesses) > 0 else 1.0


        print(f"Max Permuted Fitness: {max_perm_fitness:.6f}")
        print(f"Avg Permuted Fitness: {avg_perm_fitness:.6f}")
        print(f"Std Dev Permuted Fitness: {std_perm_fitness:.6f}")
        print(f"\nEmpirical p-value: {count_better_or_equal} / {len(all_perm_fitnesses)} = {p_value:.4f}")

        summary_lines.extend([
            f"Max Permuted Fitness: {max_perm_fitness:.6f}",
            f"Avg Permuted Fitness: {avg_perm_fitness:.6f}",
            f"Std Dev Permuted Fitness: {std_perm_fitness:.6f}",
            f"\nEmpirical p-value: {count_better_or_equal} / {len(all_perm_fitnesses)} = {p_value:.4f}"
        ])

    else:
        print("Error: Could not find any permuted run results.")
        summary_lines.append("Error: Could not find any permuted run results.")

    # --- Print Rules ---
    if top_rules_text:
        print("\n" + top_rules_text)
        summary_lines.append("\n" + top_rules_text)
    elif task0_file_found:
        print("\nTop rules section not found or empty in ga_results_0.txt.")
        summary_lines.append("\nTop rules section not found or empty in ga_results_0.txt.")
    else:
         print("\nga_results_0.txt not found, cannot display top rules.")
         summary_lines.append("\nga_results_0.txt not found, cannot display top rules.")


    # --- Save Summary ---
    summary_filename = os.path.join(results_directory, "final_summary.txt")
    try:
        with open(summary_filename, "w") as f_sum:
            f_sum.write("\n".join(summary_lines))
        print(f"\nSummary saved to: {summary_filename}")
    except Exception as e:
        print(f"\nError saving summary file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Use command-line argument for directory, or default to current directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        if not os.path.isdir(target_dir):
             print(f"Error: Provided path is not a valid directory: {target_dir}")
             sys.exit(1)
    else:
        target_dir = "." # Default to current directory

    gather_and_analyze(target_dir)


    
