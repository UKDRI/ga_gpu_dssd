import torch
import random
# import intel_extension_for_pytorch as ipex # <-- DISABLED FOR TEST
import pandas as pd
import sys
import time
import os

# --- NEW IMPORT ---
# Import the shared CPU functions
from ga_utils import get_chromosome_stats_and_fitness

# Set up device (GPU if available, fallback to CPU)
# --- MODIFIED: Forcing CPU to bypass the ipex ImportError ---
device = torch.device("cpu")
# device = torch.device("xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")


# =================== PARAMETERS =====================
# --- Read SLURM Array Task ID, Dirs, and Time Limit ---
if len(sys.argv) > 4: # Expecting script name, task_id, base_dir, output_dir, time_mins
    try:
        task_id = int(sys.argv[1])
        base_dir = sys.argv[2]
        output_dir = sys.argv[3]
        task_time_limit_mins = int(sys.argv[4])
    except Exception as e:
        print(f"Error: Could not parse all arguments: {e}")
        print("Usage: python simple_ga_array.py <task_id> <base_dir> <output_dir> <time_limit_mins>")
        sys.exit(1)
else:
    print("No task ID/paths/time provided. Running as single test (Task 0), using current dir, 10 min limit.")
    task_id = 0
    base_dir = "."
    output_dir = "."
    task_time_limit_mins = 10 # Default to 10 min for local test

print(f"--- This is SLURM Array Task ID: {task_id} ---")
print(f"--- Base directory: {base_dir} ---")
print(f"--- Saving results to directory: {output_dir} ---")
print(f"--- Job Time Limit (Minutes): {task_time_limit_mins} ---")


# --- Data/Run Parameters ---
num_permutations_per_task = 10
num_features = 1000
num_generations = 300       # Generations *per restart*
# OPTIMIZED: We can now increase this, e.g., to 100 or 120, thanks to int8
population_size = 100       # Population size per GA (was 55)
IS_REAL_DATA_RUN = (task_id == 0)

print(f"--- Using Population Size: {population_size} ---")

# --- Chromosome Structure Parameters ---
min_rules_per_chrom = 3
max_rules_per_chrom = 6
gene_inclusion_probability = 0.01

# --- (AUTO-UPDATED) ---
rule_length = 2 * num_features
max_total_bits = rule_length * max_rules_per_chrom
chromosome_length = 8 + max_total_bits

# --- GA Operator Parameters ---
tournament_size = 5
mutation_rate = 0.0001
elitism_count = 2

# --- Store all params for logging ---
PARAMS = {
    "task_id": task_id,
    "time_limit_mins": task_time_limit_mins,
    "num_permutations_per_task": num_permutations_per_task,
    "num_features": num_features,
    "num_generations_per_restart": num_generations,
    "population_size": population_size,
    "min_rules": min_rules_per_chrom,
    "max_rules": max_rules_per_chrom,
    "gene_inclusion_prob": gene_inclusion_probability,
    "tournament_size": tournament_size,
    "mutation_rate": mutation_rate,
    "elitism_count": elitism_count,
    "device": str(device),
    # --- NEW PARAMS FOR UTILS ---
    "num_features": num_features,
    "rule_length": rule_length
}


# ================== DATA MATRIX (REAL DATA) =====================
print("\nLoading real data from CSV files...")
data_dir_relative = "sc_data"
x_file = os.path.join(base_dir, data_dir_relative, "X_binary.csv")
y_file = os.path.join(base_dir, data_dir_relative, "y_labels.csv")

print(f"Attempting to load X data from: {x_file}")
print(f"Attempting to load Y data from: {y_file}")

try:
    X_df = pd.read_csv(x_file)
    gene_names = X_df.columns.tolist()
    # OPTIMIZED: Changed dtype from torch.int32 to torch.int8
    X_real_gpu = torch.tensor(X_df.values, dtype=torch.int8, device=device)
    y_df = pd.read_csv(y_file)
    # OPTIMIZED: Changed dtype from torch.int32 to torch.int8
    y_real_gpu = torch.tensor(y_df['x'].values, dtype=torch.int8, device=device)
    
    # Keep a CPU copy for final stat calculation (will inherit int8)
    X_real_cpu = X_real_gpu.cpu()
    y_real_cpu = y_real_gpu.cpu()

except FileNotFoundError as e:
    print(f"Error: Could not find data file. {e}")
    sys.exit(1)

# --- Store global data stats for later ---
DATA_STATS = {
    "total_samples": X_real_gpu.shape[0],
    "num_positives": (y_real_gpu == 1).sum().item(),
    "num_negatives": (y_real_gpu == 0).sum().item(),
    "global_pos_fraction": (y_real_gpu == 1).float().mean().item()
}

if X_real_gpu.shape[1] != num_features:
    print(f"Error: 'num_features' ({num_features}) != data columns ({X_real_gpu.shape[1]})")
    sys.exit(1)

print(f"Data loaded successfully:")
print(f"  {DATA_STATS['total_samples']} samples ({DATA_STATS['num_positives']} positives, {DATA_STATS['num_negatives']} negatives)")
print(f"  {num_features} features (genes)")

# --- 5. Create Batches ---
# These will correctly be dtype torch.int8
X_batch = X_real_gpu.unsqueeze(0).repeat(num_permutations_per_task, 1, 1)
y_batch = y_real_gpu.unsqueeze(0).repeat(num_permutations_per_task, 1)

print("Permuting labels...")
for i in range(num_permutations_per_task):
    if IS_REAL_DATA_RUN and i == 0:
        print("This is Task 0: Keeping real labels for the first run.")
        continue
    y_batch[i] = y_batch[i, torch.randperm(DATA_STATS['total_samples'])]

# This will also be torch.int8
data_batch = torch.cat([X_batch, y_batch.unsqueeze(2)], dim=2)
print(f"Data batch shape (on GPU): {data_batch.shape}")


# ================== CHROMOSOME GENERATION =====================
def generate_chromosome(device_to_use):
    rule_count = random.randint(min_rules_per_chrom, max_rules_per_chrom)
    # OPTIMIZED: Changed dtype from torch.int32 to torch.int8
    rule_count_bits = torch.tensor(list(map(int, format(rule_count, '08b'))), dtype=torch.int8, device=device_to_use)
    all_rule_bits = []
    for _ in range(max_rules_per_chrom):
        # OPTIMIZED: Changed .int() to .to(torch.int8)
        mask_bits = (torch.rand(num_features, device=device_to_use) < gene_inclusion_probability).to(torch.int8)
        # OPTIMIZED: Changed .int() to .to(torch.int8)
        value_bits = (torch.rand(num_features, device=device_to_use) > 0.5).to(torch.int8)
        all_rule_bits.append(torch.cat([mask_bits, value_bits]))
    padded_rules = torch.cat(all_rule_bits)
    # This will be torch.int8
    return torch.cat([rule_count_bits, padded_rules])


# ========== GA Wrapper Loop Setup ==========
task_start_time = time.time()

# Set a dynamic safety margin: 1 min for jobs <= 15 min, 15 min otherwise
if task_time_limit_mins <= 15:
    safety_margin_seconds = 60 # 1 minute
else:
    safety_margin_seconds = 15 * 60 # 15 minutes
    
task_max_duration_seconds = (task_time_limit_mins * 60) - safety_margin_seconds
print(f"--- Task will run for max {task_max_duration_seconds / 3600:.2f} hours (Time Limit: {task_time_limit_mins} min, Safety: {safety_margin_seconds / 60} min) ---")

# These will track the best results found *across all restarts*
overall_best_fitness_per_perm = torch.zeros(num_permutations_per_task, device=device)
overall_best_population_final_state = None # To store the best population for decoding
run_count = 0


# ========== GA FUNCTIONS (Vectorized GPU) ===========================
def get_rule_matches(population_batch, data_batch):
    N_batch_size = population_batch.shape[0]
    # Slices will be int8
    data_features = data_batch[:, :, :num_features]
    rules_data = population_batch[:, :, 8:]
    all_rules = rules_data.reshape(N_batch_size, population_size, max_rules_per_chrom, rule_length)
    all_masks = all_rules[:, :, :, :num_features]
    all_values = all_rules[:, :, :, num_features:]
    
    # Broadcasted tensors will be int8
    expanded_masks = all_masks.unsqueeze(3)
    expanded_values = all_values.unsqueeze(3)
    expanded_data = data_features.unsqueeze(1).unsqueeze(1)
    
    # This (int8 == 0) | (int8 == int8) comparison creates the large intermediate bool tensors
    rule_applies_to_feature = (expanded_masks == 0) | (expanded_data == expanded_values)
    
    # This .all() is the main reduction step
    sample_matches_rule = rule_applies_to_feature.all(dim=-1)
    return sample_matches_rule

def calculate_fitness(population_batch, data_batch, metric='wracc'):
    N_batch_size = population_batch.shape[0]
    # OPTIMIZED: Removed .int() call, tensor is already int8
    rule_count_bits = population_batch[:, :, :8]
    # OPTIMIZED: Changed dtype from torch.int32 to torch.int8 (fits 0-128)
    powers_of_2 = 2**torch.arange(7, -1, -1, device=population_batch.device, dtype=torch.int8)
    
    # (int8 * int8) -> int8. Max sum is 255, which fits in int8.
    rule_counts = (rule_count_bits * powers_of_2).sum(dim=-1)
    
    # OPTIMIZED: Added dtype=torch.int8 (fits 0-5)
    range_tensor = torch.arange(max_rules_per_chrom, device=population_batch.device, dtype=torch.int8)
    
    # (int8 < int8) -> bool
    active_rule_mask = (range_tensor < rule_counts.unsqueeze(2))
    
    all_rule_matches = get_rule_matches(population_batch, data_batch)
    active_mask_broadcast = active_rule_mask.unsqueeze(-1)
    active_matches = all_rule_matches & active_mask_broadcast
    chromosome_cover = active_matches.any(dim=2)
    
    # OPTIMIZED: Removed .int() call, tensor is already int8
    labels_flat = data_batch[:, :, -1]
    
    tp_per_chrom = (chromosome_cover & (labels_flat.unsqueeze(1) == 1)).sum(dim=-1).float()
    total_unique_matches = chromosome_cover.sum(dim=-1).float()
    total_samples_tensor = torch.tensor(DATA_STATS['total_samples'], device=population_batch.device).float()
    
    # Global pos fraction needs to be batched
    global_pos_fraction_per_perm = (labels_flat == 1).sum(dim=-1).float() / total_samples_tensor
    global_pos_fraction = global_pos_fraction_per_perm.unsqueeze(1)
    
    eps = 1e-9
    if metric == 'precision':
        quality_of_union = tp_per_chrom / (total_unique_matches + eps)
    elif metric == 'wracc':
        local_pos_fraction = tp_per_chrom / (total_unique_matches + eps)
        subgroup_size_fraction = total_unique_matches / total_samples_tensor
        quality_of_union = subgroup_size_fraction * (local_pos_fraction - global_pos_fraction)
    else:
        raise ValueError("Unknown metric. 'wracc' or 'precision'")
        
    matches_per_rule = active_matches.sum(dim=-1).float()
    total_matches_with_duplicates = matches_per_rule.sum(dim=-1)
    overlap_ratio = total_matches_with_duplicates / (total_unique_matches + eps)
    overlap_ratio = torch.clamp(overlap_ratio, min=1.0)
    fitness = quality_of_union / overlap_ratio
    fitness = torch.nan_to_num(fitness, nan=0.0)
    return fitness

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
    
    # OPTIMIZED: C=12008, so dtype=torch.int32 is correct (not int8)
    crossover_points = torch.randint(8, C, (N, P, 1), device=parent_population.device, dtype=torch.int32)
    # OPTIMIZED: C=12008, so dtype=torch.int32 is correct
    range_tensor = torch.arange(C, device=parent_population.device, dtype=torch.int32).expand(N, P, -1)
    
    # (int32 >= int32) -> bool. 
    # OPTIMIZED: Convert bool mask to int8 for multiplication
    mask = (range_tensor >= crossover_points).to(torch.int8)
    
    # (int8 * int8) + (int8 * int8) -> int8
    child_population = (parents_1 * (1 - mask)) + (parents_2 * mask)
    return child_population

def batched_mutation(child_population):
    N, P, C = child_population.shape
    rules_len = C - 8
    mutation_shape = (N, P, rules_len)
    rand_tensor = torch.rand(mutation_shape, device=child_population.device)
    mutation_mask = (rand_tensor < mutation_rate) # bool
    
    rule_counts

