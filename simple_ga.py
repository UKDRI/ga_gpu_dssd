import torch
import random
import intel_extension_for_pytorch as ipex
import pandas as pd  # Added for loading data
import sys           # Added for writing to file

# Set up device (GPU if available, fallback to CPU)
device = torch.device("xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

# =================== PARAMETERS =====================
# --- Data/Run Parameters ---
num_permutations = 10     # Number of parallel GAs (1 real + 999 permuted)
num_features = 1000         # Matches our 1000 selected genes
num_generations = 5       # How many generations to evolve
population_size = 20       # Increased for the larger search space

# --- Chromosome Structure Parameters ---
min_rules_per_chrom = 3     # Minimum rules per chromosome
max_rules_per_chrom = 6     # Maximum rules per chromosome

# --- NEW!! ---
# This is the probability that any single gene will be 'active' in a rule's mask.
# 0.01 * 1000 features = an average of 10 genes per rule.
# You can tune this! 0.02 would be 20 genes, 0.005 would be 5 genes.
gene_inclusion_probability = 0.01

# --- (AUTO-UPDATED) ---
rule_length = 2 * num_features
max_total_bits = rule_length * max_rules_per_chrom
chromosome_length = 8 + max_total_bits 

# --- GA Operator Parameters ---
tournament_size = 5         # For selection: size of the selection tournament

# --- UPDATED!! ---
# A mutation rate should be ~1.0 / (number of bits).
# Our rule_bits = 12,000. So 1/12000 is ~0.00008.
# Let's set it to 0.0001, which will flip ~1-2 bits per chromosome.
mutation_rate = 0.0001
elitism_count = 2           # Keep top 2 individuals

# ================== DATA MATRIX (REAL DATA) =====================
print("\nLoading real data from CSV files...")

# --- 1. Define file paths ---
data_dir = "sc_data"
x_file = f"{data_dir}/X_binary.csv"
y_file = f"{data_dir}/y_labels.csv"

# --- 2. Load Features (X) and Gene Names ---
try:
    # Load the features and capture the column headers (gene names)
    X_df = pd.read_csv(x_file)
    gene_names = X_df.columns.tolist() # This is our list of 1000 gene names
    
    # Convert feature data to a tensor
    X_real = torch.tensor(X_df.values, dtype=torch.int32, device=device)
    
    # --- 3. Load Labels (y) ---
    y_df = pd.read_csv(y_file)
    # NEW line
    y_real = torch.tensor(y_df['x'].values, dtype=torch.int32, device=device)

except FileNotFoundError as e:
    print(f"Error: Could not find data file.")
    print(f"Details: {e}")
    print("Please make sure 'X_binary.csv' and 'y_labels.csv' are in the 'sc_data' directory.")
    sys.exit(1) # Exit the script if data isn't found


# --- 4. Get Data Statistics ---
total_samples = X_real.shape[0]
num_positives = (y_real == 1).sum().item()
num_negatives = total_samples - num_positives

# Check that num_features matches the data
if X_real.shape[1] != num_features:
    print(f"Error: Parameter 'num_features' ({num_features}) does not match data columns ({X_real.shape[1]})")
    sys.exit(1)

print(f"Data loaded successfully:")
print(f"  {total_samples} samples ({num_positives} positives, {num_negatives} negatives)")
print(f"  {num_features} features (genes)")

# --- 5. Create Batches (Same as before) ---
X_batch = X_real.unsqueeze(0).repeat(num_permutations, 1, 1) # Shape: [1000, 6385, 1000]
y_batch = y_real.unsqueeze(0).repeat(num_permutations, 1)   # Shape: [1000, 6385]

# Permute the labels for all batches *except* the first one (index 0)
print("Permuting labels for null distribution...")
for i in range(1, num_permutations):
    y_batch[i] = y_batch[i, torch.randperm(total_samples)]

# Combine features and labels into one batched data tensor
# Shape: [1000, 6385, 1001]
data_batch = torch.cat([X_batch, y_batch.unsqueeze(2)], dim=2)

print(f"Data batch shape: {data_batch.shape}")





# ================== CHROMOSOMES =====================
# Generate one chromosome with random number of rules
def generate_chromosome():
    rule_count = random.randint(min_rules_per_chrom, max_rules_per_chrom)
    rule_count_bits = torch.tensor(list(map(int, format(rule_count, '08b'))), dtype=torch.int32, device=device)

    all_rule_bits = []
    
    # Generate each rule one by one with sparsity
    for _ in range(max_rules_per_chrom):
        # Create a SPARSE mask
        # Use gene_inclusion_probability (e.g., 1%)
        mask_bits = (torch.rand(num_features, device=device) < gene_inclusion_probability).int()
        
        # Create standard 50/50 values
        value_bits = (torch.rand(num_features, device=device) > 0.5).int()
        
        all_rule_bits.append(torch.cat([mask_bits, value_bits]))

    # We still pad the chromosome, but we only use the first 'rule_count' rules
    # This is handled by the fitness function, which respects the rule_count
    padded_rules = torch.cat(all_rule_bits)
    
    return torch.cat([rule_count_bits, padded_rules])


# Generate ONE population
base_population = torch.stack([generate_chromosome() for _ in range(population_size)])

# Create the batched population by repeating the base population
# Shape: [1000, 100, chromosome_length]
population_batch = base_population.unsqueeze(0).repeat(num_permutations, 1, 1)

print(f"\nPopulation batch shape: {population_batch.shape}")

# ========== DECODING ===========================
# This function now uses the global 'gene_names' list loaded from the CSV
def decode_rule(rule_tensor):
    mask = rule_tensor[:num_features]
    values = rule_tensor[num_features:]
    
    conditions = []
    for i in range(num_features):
        if mask[i].item() == 1:
            # Use the actual gene name!
            gene = gene_names[i]
            val = values[i].item()
            conditions.append(f"{gene}={val}")
            
    return " AND ".join(conditions) if conditions else "Always true"

def decode_chromosome(chromosome):
    rule_count_bin = chromosome[:8].tolist()
    rule_count = int("".join(map(str, rule_count_bin)), 2)
    decoded_rules = []
    for i in range(rule_count):
        start = 8 + i * rule_length
        end = start + rule_length
        rule_tensor = chromosome[start:end]
        decoded_rules.append(decode_rule(rule_tensor))
    return decoded_rules

# --- MODIFIED: Only print one chromosome as an example ---
print(f"\nPrinting 1 of {population_size} initial chromosomes (format example):")

# Get the first chromosome to print
chrom_to_print = base_population[0]
rule_count_bin = chrom_to_print[:8].tolist()
rule_count = int("".join(map(str, rule_count_bin)), 2)

print(f"\nChromosome 1:")
print(f"  [RULE COUNT] {''.join(map(str, rule_count_bin))} ({rule_count} rules)")

# Just print the *decoded rule* for the example, it's more readable
decoded_rules_example = decode_chromosome(chrom_to_print)
for j, rule in enumerate(decoded_rules_example):
    print(f"  Rule {j+1}: {rule}")

# (This section was a duplicate and has been removed)

def get_rule_matches(population_batch, data_batch):
    """
    Calculates which data samples are matched by which rules in a batched manner.
    
    Args:
    population_batch (Tensor): Shape [num_perms, pop_size, chromosome_length]
    data_batch (Tensor): Shape [num_perms, total_samples, num_features + 1]

    Returns:
    Tensor: A boolean tensor of shape [num_perms, pop_size, max_rules, total_samples]
            True if sample `s` is matched by rule `r` of chromosome `p` 
            in permutation `n`.
    """
    
    # --- 1. Extract Tensors ---
    # Get features and labels from data
    # data_features shape: [num_perms, total_samples, num_features]
    data_features = data_batch[:, :, :num_features]
    
    # --- 2. Decode Chromosomes ---
    # Get the rules part of the chromosomes
    # rules_data shape: [num_perms, pop_size, max_total_bits]
    rules_data = population_batch[:, :, 8:]

    # Reshape into individual rules
    # all_rules shape: [num_perms, pop_size, max_rules, rule_length]
    all_rules = rules_data.reshape(num_permutations, 
                                   population_size, 
                                   max_rules_per_chrom, 
                                   rule_length)

    # Split into masks and values
    # all_masks shape: [num_perms, pop_size, max_rules, num_features]
    # all_values shape: [num_perms, pop_size, max_rules, num_features]
    all_masks = all_rules[:, :, :, :num_features]
    all_values = all_rules[:, :, :, num_features:]

    # --- 3. The "Broadcasting Magic" ---
    # all_masks:     [N, P, R, F] (N=perms, P=pop_size, R=rules, F=features)
    # all_values:    [N, P, R, F]
    # data_features: [N, S, F]    (S=samples)
    
    # [N, P, R, 1, F]
    expanded_masks = all_masks.unsqueeze(3)
    # [N, P, R, 1, F]
    expanded_values = all_values.unsqueeze(3)
    # [N, 1, 1, S, F]
    expanded_data = data_features.unsqueeze(1).unsqueeze(1)

    # rule_applies_to_feature shape: [N, P, R, S, F]
    rule_applies_to_feature = (expanded_masks == 0) | (expanded_data == expanded_values)
    
    # A sample matches a rule *only if* all feature conditions are met.
    # We .all() along the feature dimension (dim=-1)
    # sample_matches_rule shape: [N, P, R, S]
    sample_matches_rule = rule_applies_to_feature.all(dim=-1)

    return sample_matches_rule


# --- 4. A FINAL Fitness Function (Quality of Union / Overlap Ratio) ---
def calculate_fitness(population_batch, data_batch, metric='wracc'):
    """
    Calculates fitness based on the *union* of all active rules,
    divided by an overlap-penalty (overlap_ratio).
    """
    
    # --- A: Get rule_count from first 8 bits ---
    # [N, P, 8]
    rule_count_bits = population_batch[:, :, :8].int() 
    powers_of_2 = 2**torch.arange(7, -1, -1, device=device, dtype=torch.int32)
    # [N, P]
    rule_counts = (rule_count_bits * powers_of_2).sum(dim=-1)

    # --- B: Create mask for active rules ---
    range_tensor = torch.arange(max_rules_per_chrom, device=device)
    # [N, P, R]
    active_rule_mask = (range_tensor < rule_counts.unsqueeze(2))
    
    # --- C: Get Matches & Stats ---
    # Get all rule matches, shape [N, P, R, S]
    all_rule_matches = get_rule_matches(population_batch, data_batch)
    
    # Mask out matches from *inactive* rules
    # [N, P, R, 1]
    active_mask_broadcast = active_rule_mask.unsqueeze(-1)
    
    # [N, P, R, S]
    active_matches = all_rule_matches & active_mask_broadcast
    
    # --- D: Calculate the UNION of all active rule covers ---
    # [N, P, S]
    chromosome_cover = active_matches.any(dim=2)

    # --- E: Calculate Quality of this single UNION cover (Exploitation) ---
    
    # Get all labels, shape [N, S]
    labels_flat = data_batch[:, :, -1].int()
    
    # 1. Calculate True Positives (TP) for the union cover
    # [N, P]
    tp_per_chrom = (chromosome_cover & (labels_flat.unsqueeze(1) == 1)).sum(dim=-1).float()
    
    # 2. Calculate Total Matches for the union cover
    # [N, P]
    total_unique_matches = chromosome_cover.sum(dim=-1).float()
    
    # 3. Calculate global dataset stats
    # Use total_samples variable defined in the data loading section
    total_samples_tensor = torch.tensor(total_samples, device=device).float()
    
    # [N] -> [N, 1]
    global_pos_fraction = (labels_flat == 1).sum(dim=-1).float() / total_samples_tensor
    global_pos_fraction = global_pos_fraction.unsqueeze(1)
    
    eps = 1e-9
    
    # 4. Calculate quality (e.g., WRAcc) of the union
    if metric == 'precision':
        quality_of_union = tp_per_chrom / (total_unique_matches + eps)
        
    elif metric == 'wracc':
        local_pos_fraction = tp_per_chrom / (total_unique_matches + eps)
        subgroup_size_fraction = total_unique_matches / total_samples_tensor
        quality_of_union = subgroup_size_fraction * (local_pos_fraction - global_pos_fraction)
        
    elif metric == 'wkl':
        p_S_1 = global_pos_fraction
        p_G_1 = tp_per_chrom / (total_unique_matches + eps)
        p_S_0 = 1.0 - p_S_1
        p_G_0 = 1.0 - p_G_1
        term1 = p_G_1 * torch.log2((p_G_1 / (p_S_1 + eps)) + eps)
        term2 = p_G_0 * torch.log2((p_G_0 / (p_S_0 + eps)) + eps)
        kl_divergence = term1 + term2
        subgroup_size_fraction = total_unique_matches / total_samples_tensor
        quality_of_union = subgroup_size_fraction * kl_divergence
        
    else:
        raise ValueError("Unknown metric. Choose 'precision', 'wracc', or 'wkl'")

    # --- F: Calculate Diversity Penalty (Exploration) ---
    
    # 1. Sum of all matches *including duplicates*
    # [N, P, R]
    matches_per_rule = active_matches.sum(dim=-1).float()
    # [N, P]
    total_matches_with_duplicates = matches_per_rule.sum(dim=-1)
    
    # 2. Calculate Overlap Ratio
    # [N, P]
    overlap_ratio = total_matches_with_duplicates / (total_unique_matches + eps)
    
    # Clamp ratio at 1.0
    overlap_ratio = torch.clamp(overlap_ratio, min=1.0)

    # --- G: Calculate Final Fitness ---
    fitness = quality_of_union / overlap_ratio
    
    # Handle NaNs that might arise from 0/0 divisions
    fitness = torch.nan_to_num(fitness, nan=0.0)
    
    return fitness


def batched_selection(population_batch, fitness_scores):
    """
    Performs batched tournament selection.
    """
    N, P, C = population_batch.shape
    
    # [N, P, T] (T=tournament_size)
    tournament_indices = torch.randint(0, P, (N, P, tournament_size), device=device)
    
    # [N, P] -> [N, P, 1] -> [N, P, P]
    expanded_fitness = fitness_scores.unsqueeze(1).expand(-1, P, -1)
    
    # Gather fitnesses: [N, P, T]
    tournament_fitnesses = torch.gather(expanded_fitness, 2, tournament_indices)
    
    # Find the index *within the tournament* of the winner: [N, P]
    winner_local_indices = torch.argmax(tournament_fitnesses, dim=2)
    
    # [N, P] -> [N, P, 1]
    winner_local_indices = winner_local_indices.unsqueeze(2)
    
    # Get the global index of the winner: [N, P]
    winner_indices = torch.gather(tournament_indices, 2, winner_local_indices).squeeze(2)

    # Gather the winning chromosomes from the original population
    # [N, P, C]
    expanded_winner_indices = winner_indices.unsqueeze(2).expand(-1, -1, C)
    
    # New parent population: [N, P, C]
    parent_population = torch.gather(population_batch, 1, expanded_winner_indices)
    
    return parent_population

def batched_crossover(parent_population):
    """
    Performs batched single-point crossover.
    """
    N, P, C = parent_population.shape
    
    # Shuffle parents along the population dimension to create pairs
    indices = torch.randperm(P, device=device)
    parents_1 = parent_population
    parents_2 = parent_population[:, indices, :]
    
    # Crossover point is *after* the 8-bit rule count
    crossover_points = torch.randint(8, C, (N, P, 1), device=device)
    
    # [N, P, C] tensor of [0, 1, 2, ..., C-1]
    range_tensor = torch.arange(C, device=device).expand(N, P, -1)
    
    # Mask is 0 where range < point, 1 where range >= point
    mask = (range_tensor >= crossover_points).int()
    
    # Child 1 gets bits from Parent 1 where mask is 0, Parent 2 where mask is 1
    child_population = (parents_1 * (1 - mask)) + (parents_2 * mask)
    
    return child_population

def batched_mutation(child_population):
    """
    Performs batched bit-flip mutation, *sparing the 8-bit rule count*.
    """
    N, P, C = child_population.shape
    
    # Create a mutation mask for the *rules portion* only
    rules_len = C - 8
    mutation_shape = (N, P, rules_len)
    
    rand_tensor = torch.rand(mutation_shape, device=device)
    
    mutation_mask = (rand_tensor < mutation_rate)
    
    # Split the population
    rule_counts = child_population[:, :, :8]
    rules = child_population[:, :, 8:]
    
    # Apply mutation using logical XOR
    mutated_rules = rules.logical_xor(mutation_mask)
    
    # Re-combine
    mutated_population = torch.cat([rule_counts, mutated_rules], dim=2)
    
    return mutated_population


# =================== MAIN GA LOOP =====================

# Keep track of the best fitness found so far for each permutation
best_fitness_per_perm = torch.zeros(num_permutations, device=device)

print(f"\nStarting {num_permutations} parallel GA runs for {num_generations} generations...")

for gen in range(num_generations):
    
    # 1. Calculate Fitness
    fitness_scores = calculate_fitness(population_batch, data_batch)
    
    # --- Elitism ---
    best_fitnesses, best_indices = torch.topk(fitness_scores, elitism_count, dim=1)
    
    # [N, elitism_count, C]
    elite_individuals = torch.gather(
        population_batch, 1, 
        best_indices.unsqueeze(2).expand(-1, -1, chromosome_length)
    )
    
    # Update our overall best fitness tracker
    best_fitness_per_perm = torch.max(best_fitness_per_perm, best_fitnesses[:, 0])

    # 2. Selection
    parent_population = batched_selection(population_batch, fitness_scores)
    
    # 3. Crossover
    child_population = batched_crossover(parent_population)
    
    # 4. Mutation
    mutated_population = batched_mutation(child_population)
    
    # --- Replace Population ---
    mutated_population[:, :elitism_count, :] = elite_individuals
    
    population_batch = mutated_population
    
    if (gen + 1) % 10 == 0:
        print(f"  Gen {gen+1:3d}: "
              f"Real data best fitness: {best_fitness_per_perm[0].item():.4f}, "
              f"Avg permuted best fitness: {best_fitness_per_perm[1:].mean().item():.4f}")

print("\n...Evolution complete.")

# =================== FINAL RESULTS (writes to file) =====================

print("Calculating final results and writing to 'ga_results.txt'...")

# Open the output file
with open("ga_results.txt", "w") as f:

    # Define how many of the top rule sets we want to see
    num_top_sets_to_show = 5 # Increased to 5

    # Get final fitness scores
    final_fitness_scores = calculate_fitness(population_batch, data_batch)

    # Get the top N best fitnesses and their indices for *all* permutations
    final_best_fitnesses, final_best_indices = torch.topk(
        final_fitness_scores, num_top_sets_to_show, dim=1
    )

    # --- Print Top N Rule Sets from REAL Data (Permutation 0) ---
    f.write(f"Top {num_top_sets_to_show} rule sets from REAL data:\n")
    print(f"\nTop {num_top_sets_to_show} rule sets from REAL data:")

    for i in range(num_top_sets_to_show):
        chrom_index = final_best_indices[0, i]
        fitness = final_fitness_scores[0, chrom_index].item()
        chromosome = population_batch[0, chrom_index, :]

        output_str = f"\n--- Rank {i+1} (Fitness: {fitness:.4f}) ---"
        f.write(output_str + "\n")
        print(output_str)

        rules = decode_chromosome(chromosome)
        
        if not rules:
            f.write("  (No active rules found)\n")
            print("  (No active rules found)")
        
        for j, rule in enumerate(rules):
            rule_str = f"  Rule {j+1}: {rule}"
            f.write(rule_str + "\n")
            print(rule_str)

    # --- Statistical Test ---
    real_fitness = final_best_fitnesses[0, 0].item()     # Best fitness from real run (Rank 1)
    permuted_fitnesses = final_best_fitnesses[1:, 0]  # Best fitness from each permuted run

    p_value = (permuted_fitnesses >= real_fitness).float().mean().item()
    avg_permuted_fitness = permuted_fitnesses.mean().item()
    max_permuted_fitness = permuted_fitnesses.max().item()

    f.write(f"\nPermutation Test:\n")
    f.write(f"  Best real data fitness (Rank 1): {real_fitness:.4f}\n")
    f.write(f"  Max fitness in permuted runs: {max_permuted_fitness:.4f}\n")
    f.write(f"  Avg best null fitness (from permuted runs): {avg_permuted_fitness:.4f}\n")
    f.write(f"  p-value (empirical): {p_value:.4f}\n")
    
    print(f"\nPermutation Test:")
    print(f"  Best real data fitness (Rank 1): {real_fitness:.4f}")
    print(f"  Max fitness in permuted runs: {max_permuted_fitness:.2f}")
    print(f"  Avg best null fitness (from permuted runs): {avg_permuted_fitness:.4f}")
    print(f"  p-value (empirical): {p_value:.4f}")

    # --- New Population Analysis ---
    real_run_all_fitnesses = final_fitness_scores[0, :] # Shape [P]
    count_better_than_null_avg = (real_run_all_fitnesses > avg_permuted_fitness).sum().item()

    f.write(f"\nFinal Population Analysis (Real Data):\n")
    f.write(f"  {count_better_than_null_avg} of {population_size} chromosomes had fitness > avg. null fitness ({avg_permuted_fitness:.4f})\n")

    print(f"\nFinal Population Analysis (Real Data):")
    print(f"  {count_better_than_null_avg} of {population_size} chromosomes had fitness > avg. null fitness ({avg_permuted_fitness:.4f})")

print("\nDone. Results saved to 'ga_results.txt'.")
