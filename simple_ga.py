import torch
import random
# Check for intel_extension_for_pytorch (ipex)
try:
    import intel_extension_for_pytorch as ipex
    print("Intel Extension for PyTorch (ipex) found.")
except ImportError:
    ipex = None
    print("Intel Extension for PyTorch (ipex) not found. Will use standard CUDA or CPU.")

import pandas as pd
import sys
import time
import os
import gc

# --- Device Setup ---
if ipex and hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu:0")
    print("Using device: XPU (Intel GPU)")
    torch.xpu.set_device(device)
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using device: CUDA (NVIDIA GPU)")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("Using device: CPU")


# =================== PARAMETERS =====================
# --- Read SLURM Args ---
if len(sys.argv) > 3:
    try:
        task_id = int(sys.argv[1])
        base_dir = sys.argv[2]
        output_dir = sys.argv[3]
    except (ValueError, IndexError):
         print("Usage: python simple_ga_array.py <task_id> <base_dir> <output_dir>")
         sys.exit(1)
else:
    print("No task ID/paths provided. Running as a single test (Task 0), using current dir.")
    task_id = 0
    base_dir = "."
    output_dir = "." # Default to current directory

print(f"--- Task ID: {task_id}, Base Dir: {base_dir}, Output Dir: {output_dir} ---")

# --- GA Parameters ---
# This script handles ONE permutation (real or permuted)
# It runs the GA num_repeats times to find the best result for this single permutation.
num_repeats = 200      # Number of GA runs per permutation
num_generations = 1000 # Generations per single GA run
population_size = 55   # Max feasible population

num_features = 1000
IS_REAL_DATA_RUN = (task_id == 0)

# --- Chromosome Structure ---
min_rules_per_chrom = 3
max_rules_per_chrom = 6
gene_inclusion_probability = 0.01
rule_length = 2 * num_features
max_total_bits = rule_length * max_rules_per_chrom
chromosome_length = 8 + max_total_bits

# --- GA Operators ---
tournament_size = 5
mutation_rate = 0.0001
elitism_count = 2


# ================== DATA LOADING =====================
print("\nLoading real data...")
data_dir_relative = "sc_data"
x_file = os.path.join(base_dir, data_dir_relative, "X_binary.csv")
y_file = os.path.join(base_dir, data_dir_relative, "y_labels.csv")

try:
    X_df = pd.read_csv(x_file)
    gene_names = X_df.columns.tolist() # Keep gene names in memory
    X_real_cpu = torch.tensor(X_df.values, dtype=torch.int32)
    y_df = pd.read_csv(y_file)
    y_real_cpu = torch.tensor(y_df['x'].values, dtype=torch.int32)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print(f"Looked for: {x_file} and {y_file}")
    sys.exit(1)
except KeyError:
    print("Error: 'x' column not found in y_labels.csv. Make sure it was saved correctly.")
    sys.exit(1)

total_samples = X_real_cpu.shape[0]
print(f"Data loaded: {total_samples} samples, {num_features} features")
del X_df, y_df # Free memory


# ================== PREPARE THIS TASK'S DATA =====================
# This task processes ONE permutation, but searches it num_repeats times.
# We create a data_batch of size 1 (N=1)

print(f"Preparing data for Task {task_id}...")
if IS_REAL_DATA_RUN:
    print("Task 0: Using REAL labels.")
    y_perm_cpu = y_real_cpu
else:
    print(f"Task {task_id}: Creating permuted labels.")
    # Ensure reproducibility if needed, but for permutation test, random is fine
    # torch.manual_seed(task_id) 
    y_perm_cpu = y_real_cpu[torch.randperm(total_samples)]

# Create the N=1 data batch on the GPU
try:
    data_batch_gpu = torch.cat([X_real_cpu, y_perm_cpu.unsqueeze(1)], dim=1).unsqueeze(0).to(device)
    print(f"Data batch created on GPU: {data_batch_gpu.shape}")
except Exception as e:
    print(f"Error moving data batch to GPU: {e}")
    sys.exit(1)


del X_real_cpu, y_real_cpu, y_perm_cpu # Free CPU memory
gc.collect()


# ================== HELPER FUNCTIONS =====================
def clear_gpu_cache():
    """Helper to clear GPU cache based on device type."""
    if device.type == 'xpu':
        torch.xpu.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

def generate_chromosome(device_to_use):
    rule_count = random.randint(min_rules_per_chrom, max_rules_per_chrom)
    rule_count_bits = torch.tensor(list(map(int, format(rule_count, '08b'))), dtype=torch.int32, device=device_to_use)
    all_rule_bits = []
    for _ in range(max_rules_per_chrom):
        mask_bits = (torch.rand(num_features, device=device_to_use) < gene_inclusion_probability).int()
        value_bits = (torch.rand(num_features, device=device_to_use) > 0.5).int()
        all_rule_bits.append(torch.cat([mask_bits, value_bits]))
    padded_rules = torch.cat(all_rule_bits)
    return torch.cat([rule_count_bits, padded_rules])

def decode_rule(rule_tensor): # Assumes rule_tensor is on CPU
    mask = rule_tensor[:num_features]
    values = rule_tensor[num_features:]
    conditions = []
    for i in range(num_features):
        if mask[i].item() == 1:
            gene = gene_names[i]
            val = values[i].item()
            conditions.append(f"{gene}={val}")
    return " AND ".join(conditions) if conditions else "Always true"

def decode_chromosome(chromosome): # Assumes chromosome is on CPU
    chromosome_cpu = chromosome
    rule_count_bin = chromosome_cpu[:8].tolist()
    rule_count = int("".join(map(str, rule_count_bin)), 2)
    decoded_rules = []
    for i in range(rule_count):
        start = 8 + i * rule_length
        end = start + rule_length
        rule_tensor_cpu = chromosome_cpu[start:end]
        decoded_rules.append(decode_rule(rule_tensor_cpu))
    return decoded_rules

def get_rule_matches(population_batch, data_batch):
    N_batch_size = population_batch.shape[0] # N will be 1
    data_features = data_batch[:, :, :num_features]
    rules_data = population_batch[:, :, 8:]
    
    try:
        all_rules = rules_data.reshape(N_batch_size, population_size, max_rules_per_chrom, rule_length)
        all_masks = all_rules[:, :, :, :num_features]
        all_values = all_rules[:, :, :, num_features:]
        
        expanded_masks = all_masks.unsqueeze(3)
        expanded_values = all_values.unsqueeze(3)
        expanded_data = data_features.unsqueeze(1).unsqueeze(1)
        
        rule_applies_to_feature = (expanded_masks == 0) | (expanded_data == expanded_values)
        sample_matches_rule = rule_applies_to_feature.all(dim=-1)
        
        del expanded_masks, expanded_values, expanded_data, rule_applies_to_feature
        del all_rules, all_masks, all_values, rules_data, data_features
        return sample_matches_rule
        
    except Exception as e:
        print(f"Error in get_rule_matches: {e}")
        clear_gpu_cache()
        gc.collect()
        # Return an empty tensor or re-raise
        raise e

def calculate_fitness(population_batch, data_batch, metric='wracc'):
    N_batch_size = population_batch.shape[0] # N will be 1
    
    try:
        rule_count_bits = population_batch[:, :, :8].int()
        powers_of_2 = 2**torch.arange(7, -1, -1, device=population_batch.device, dtype=torch.int32)
        rule_counts = (rule_count_bits * powers_of_2).sum(dim=-1)
        range_tensor = torch.arange(max_rules_per_chrom, device=population_batch.device)
        active_rule_mask = (range_tensor < rule_counts.unsqueeze(2))
        
        all_rule_matches = get_rule_matches(population_batch, data_batch)
        
        active_mask_broadcast = active_rule_mask.unsqueeze(-1)
        active_matches = all_rule_matches & active_mask_broadcast
        chromosome_cover = active_matches.any(dim=2)
        labels_flat = data_batch[:, :, -1].int()
        
        tp_per_chrom = (chromosome_cover & (labels_flat.unsqueeze(1) == 1)).sum(dim=-1).float()
        total_unique_matches = chromosome_cover.sum(dim=-1).float()
        total_samples_tensor = torch.tensor(total_samples, device=population_batch.device).float()
        global_pos_fraction = (labels_flat == 1).sum(dim=-1).float() / total_samples_tensor
        global_pos_fraction = global_pos_fraction.unsqueeze(1)
        
        eps = 1e-9
        if metric == 'wracc':
            local_pos_fraction = tp_per_chrom / (total_unique_matches + eps)
            subgroup_size_fraction = total_unique_matches / total_samples_tensor
            quality_of_union = subgroup_size_fraction * (local_pos_fraction - global_pos_fraction)
        else: 
            raise ValueError("Only 'wracc' supported")
            
        matches_per_rule = active_matches.sum(dim=-1).float()
        total_matches_with_duplicates = matches_per_rule.sum(dim=-1)
        overlap_ratio = total_matches_with_duplicates / (total_unique_matches + eps)
        overlap_ratio = torch.clamp(overlap_ratio, min=1.0)
        
        fitness = quality_of_union / overlap_ratio
        fitness = torch.nan_to_num(fitness, nan=0.0)
        
        del rule_count_bits, rule_counts, active_rule_mask, all_rule_matches, active_matches
        del chromosome_cover, labels_flat, tp_per_chrom, total_unique_matches, quality_of_union
        del matches_per_rule, total_matches_with_duplicates, overlap_ratio
        return fitness
        
    except Exception as e:
        print(f"Error in calculate_fitness: {e}")
        clear_gpu_cache()
        gc.collect()
        raise e

def batched_selection(population_batch, fitness_scores):
    N, P, C = population_batch.shape
    tournament_indices = torch.randint(0, P, (N, P, tournament_size), device=population_batch.device)
    expanded_fitness = fitness_scores.unsqueeze(1).expand(-1, P, -1)
    tournament_fitnesses = torch.gather(expanded_fitness, 2, tournament_indices)
    winner_local_indices = torch.argmax(tournament_fitnesses, dim=2)
    winner_local_indices = winner_local_indices.unsqueeze(2)
    winner_indices = torch.gather(tournament_indices, 2, winner_local_indices).squeeze(2)
    expanded_winner_indices = winner_indices.unsqueeze(2).expand(-1, -1, C)
    parent_population = torch.gather(population_batch, 1, expanded_winner_indices)
    return parent_population

def batched_crossover(parent_population):
    N, P, C = parent_population.shape
    indices = torch.randperm(P, device=parent_population.device)
    parents_1 = parent_population
    parents_2 = parent_population[:, indices, :]
    crossover_points = torch.randint(8, C, (N, P, 1), device=parent_population.device)
    range_tensor = torch.arange(C, device=parent_population.device).expand(N, P, -1)
    mask = (range_tensor >= crossover_points).int()
    child_population = (parents_1 * (1 - mask)) + (parents_2 * mask)
    return child_population

def batched_mutation(child_population):
    N, P, C = child_population.shape
    rules_len = C - 8
    mutation_shape = (N, P, rules_len)
    rand_tensor = torch.rand(mutation_shape, device=child_population.device)
    mutation_mask = (rand_tensor < mutation_rate)
    rule_counts = child_population[:, :, :8]
    rules = child_population[:, :, 8:]
    mutated_rules = rules.logical_xor(mutation_mask)
    mutated_population = torch.cat([rule_counts, mutated_rules], dim=2)
    return mutated_population

# ================== MAIN GA RUN FUNCTION (SINGLE PERMUTATION) =====================
def run_ga(data_batch_gpu, num_gens, pop_size):
    """
    Runs the GA for specified generations on a N=1 data_batch.
    Returns:
        best_fitness_found (float): The single best fitness value found across all generations.
        best_chromosome (Tensor): The chromosome (on CPU) that achieved the best fitness.
    """
    num_perms_in_batch = data_batch_gpu.shape[0] # This must be 1
    if num_perms_in_batch != 1:
        print(f"Error: run_ga function expected N=1, but got N={num_perms_in_batch}")
        return -1.0, None

    try:
        base_pop = torch.stack([generate_chromosome(device_to_use=device) for _ in range(pop_size)])
        pop_batch = base_pop.unsqueeze(0) # Shape [1, P, C]
        
        best_fitness_value = -float('inf')
        best_chromosome_ever = None # Store the best chromosome
        
        for gen in range(num_gens):
            fitness_scores = calculate_fitness(pop_batch, data_batch_gpu) # fitness_scores shape [1, P]
            best_fitnesses_gen, best_indices_gen = torch.topk(fitness_scores, elitism_count, dim=1) # shape [1, elitism]
            elite_individuals = torch.gather(pop_batch, 1, best_indices_gen.unsqueeze(2).expand(-1, -1, chromosome_length))
            
            # Check if new best *overall* fitness is found
            current_gen_best_fitness = best_fitnesses_gen[0, 0].item()
            if current_gen_best_fitness > best_fitness_value:
                best_fitness_value = current_gen_best_fitness
                # Store the best chromosome (on CPU)
                best_chromosome_ever = pop_batch[0, best_indices_gen[0, 0], :].cpu().clone()
            
            parents = batched_selection(pop_batch, fitness_scores)
            children = batched_crossover(parents)
            mutants = batched_mutation(children)
            mutants[:, :elitism_count, :] = elite_individuals
            pop_batch = mutants

            # --- Memory Cleanup ---
            del fitness_scores, best_fitnesses_gen, best_indices_gen, elite_individuals
            del parents, children, mutants
            clear_gpu_cache()
            gc.collect()

            if (gen + 1) % 100 == 0:
                print(f"    Gen {gen+1:4d}: Best Fitness: {best_fitness_value:.6f}")
        
        # Return the single best fitness value and the best chromosome (on CPU)
        return best_fitness_value, best_chromosome_ever

    except Exception as e:
        print(f"!!! ERROR during GA run: {e}")
        clear_gpu_cache()
        gc.collect()
        return -1.0, None # Return error values
    finally:
        # Final cleanup for this GA run
        if 'pop_batch' in locals(): del pop_batch
        if 'base_pop' in locals(): del base_pop
        clear_gpu_cache()
        gc.collect()


# =================== MAIN EXECUTION LOOP (REPEATS) =====================
task_overall_best_fitness = -1.0
task_overall_best_chromosome = None

start_time_total = time.time()
print(f"\n--- Task {task_id}: Starting {num_repeats} GA repeats ({num_generations} gens, {population_size} pop) ---")

for run_num in range(num_repeats):
    print(f"\n--- Repeat {run_num + 1} of {num_repeats} ---")
    repeat_start_time = time.time()
    
    # Run the GA on this task's single data batch
    fitness_this_repeat, chromosome_this_repeat = run_ga(data_batch_gpu, num_generations, population_size)
    
    print(f"Repeat {run_num+1}: Best Fitness: {fitness_this_repeat:.6f}")

    # Update task's overall best
    if fitness_this_repeat > task_overall_best_fitness:
        print(f"  >>> New best fitness for Task {task_id}! Previous best: {task_overall_best_fitness:.6f}")
        task_overall_best_fitness = fitness_this_repeat
        task_overall_best_chromosome = chromosome_this_repeat # Already on CPU
        
    repeat_end_time = time.time()
    print(f"Repeat {run_num+1} complete. Time: {repeat_end_time - repeat_start_time:.2f} seconds.")
    
    del fitness_this_repeat, chromosome_this_repeat
    clear_gpu_cache()
    gc.collect()

total_end_time = time.time()
print(f"\n--- Task {task_id} Finished All Repeats ---")
print(f"Total task time: {total_end_time - start_time_total:.2f} seconds.")
print(f"Best fitness found for this permutation (Task {task_id}): {task_overall_best_fitness:.6f}")


# =================== FINAL RESULTS =====================
output_filename = os.path.join(output_dir, f"ga_results_{task_id}.txt")
print(f"Writing final results to '{output_filename}'...")

try:
    with open(output_filename, "w") as f:
        # --- Write the SINGLE best fitness from all repeats ---
        f.write("--- Best Fitness (Best-of-N-Repeats) ---\n")
        f.write(f"{task_overall_best_fitness:.6f}\n")
        
        # --- Write if this was the real data run ---
        f.write("\n--- Is Real Data Run ---\n")
        f.write(f"{IS_REAL_DATA_RUN}\n")

        # --- If this is Task 0, save the best chromosome and rules ---
        if IS_REAL_DATA_RUN and task_overall_best_chromosome is not None:
            f.write("\n--- Best Real Chromosome (Binary) ---\n")
            binary_string = "".join(map(str, task_overall_best_chromosome.tolist()))
            f.write(binary_string + "\n")

            f.write("\n--- Decoded Top Rules from Best Real Chromosome ---\n")
            try:
                decoded_rules = decode_chromosome(task_overall_best_chromosome)
                if not decoded_rules:
                    f.write("  (No active rules found)\n")
                else:
                    active_rules_count = 0
                    for j, rule in enumerate(decoded_rules[:max_rules_per_chrom]):
                        if rule != "Always true":
                            rule_str = f"  Rule {j+1}: {rule}"
                            f.write(rule_str + "\n")
                            active_rules_count += 1
                    if active_rules_count == 0:
                        f.write("  (All rules evaluated to 'Always true')\n")
            except Exception as e:
                f.write(f"  Error decoding chromosome: {e}\n")
                print(f"Error decoding best chromosome for saving: {e}")
        elif IS_REAL_DATA_RUN:
            f.write("\n--- No valid real chromosome found by Task 0 ---\n")

except Exception as e:
    print(f"!!! CRITICAL ERROR: Failed to write results file: {e}")
    # Try to write a minimal error file
    try:
        error_filename = os.path.join(output_dir, f"ga_ERROR_task_{task_id}.txt")
        with open(error_filename, "w") as f_err:
            f_err.write(f"Task {task_id} failed to write final results.\n")
            f_err.write(f"Error: {e}\n")
            f_err.write(f"Best fitness found before crash: {task_overall_best_fitness}\n")
    except:
        pass # Final attempt failed

print(f"\nDone. Results saved to '{output_filename}'.")

