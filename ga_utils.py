import torch
import re

# This file contains CPU-based utility functions shared between
# the GA runner (simple_ga_array.py) and the analysis script (gather_results.py).

def decode_rule_cpu(rule_tensor_cpu, gene_names_list, num_features):
    """Decodes a single CPU rule tensor into a string."""
    mask = rule_tensor_cpu[:num_features]
    values = rule_tensor_cpu[num_features:]
    conditions = []
    for i in range(num_features):
        if mask[i].item() == 1:
            gene = gene_names_list[i]
            val = values[i].item()
            conditions.append(f"{gene}={val}")
    return " AND ".join(conditions) if conditions else "Always true"

def get_rule_matches_cpu(rule_tensor_cpu, X_data_cpu, num_features):
    """Finds which samples match a single CPU rule tensor."""
    mask = rule_tensor_cpu[:num_features].bool()
    values = rule_tensor_cpu[num_features:]
    # Samples match if (feature is not in mask) OR (feature value == rule value)
    rule_applies_to_feature = (~mask) | (X_data_cpu == values)
    # A sample matches the rule if it matches all features
    sample_matches_rule = rule_applies_to_feature.all(dim=1)
    return sample_matches_rule

def get_chromosome_stats_and_fitness(chromosome_cpu, X_data_cpu, y_data_cpu, gene_names_list, global_stats, param_dict):
    """
    Calculates detailed stats and fitness for a SINGLE chromosome on the CPU.
    This is for final reporting, not for GA evolution.
    """
    try:
        # Extract params
        num_features = param_dict['num_features']
        rule_length = 2 * num_features
        
        eps = 1e-9
        rule_count_bin = chromosome_cpu[:8].tolist()
        rule_count = int("".join(map(str, rule_count_bin)), 2)
        
        rule_stats = []
        all_rule_matches_masks = []
        total_matches_with_duplicates = 0

        for i in range(rule_count):
            start = 8 + i * rule_length
            end = start + rule_length
            rule_tensor_cpu = chromosome_cpu[start:end]
            
            decoded_rule_str = decode_rule_cpu(rule_tensor_cpu, gene_names_list, num_features)
            if decoded_rule_str == "Always true":
                rule_stats.append({
                    "rule": decoded_rule_str,
                    "n_matched": 0, "n_pos": 0, "n_neg": 0
                })
                continue # 'Always true' rules are ignored

            matches_mask = get_rule_matches_cpu(rule_tensor_cpu, X_data_cpu, num_features) # Bool tensor [n_samples]
            all_rule_matches_masks.append(matches_mask)
            
            n_matched = matches_mask.sum().item()
            total_matches_with_duplicates += n_matched
            
            # Find positives (pos=1) and negatives (pos=0) covered
            n_pos = (matches_mask & (y_data_cpu == 1)).sum().item()
            n_neg = (matches_mask & (y_data_cpu == 0)).sum().item()
            
            rule_stats.append({
                "rule": decoded_rule_str,
                "n_matched": n_matched,
                "n_pos": n_pos,
                "n_neg": n_neg
            })

        if not all_rule_matches_masks: # No active rules found
            return {"error": "No active rules."}

        # --- Chromosome (Rule Set) Stats ---
        # Find the *union* of all rule matches
        chromosome_cover_mask = torch.stack(all_rule_matches_masks).any(dim=0)
        
        N_total_unique_matches = chromosome_cover_mask.sum().item()
        N_tp = (chromosome_cover_mask & (y_data_cpu == 1)).sum().item() # True Positives
        N_fp = (chromosome_cover_mask & (y_data_cpu == 0)).sum().item() # False Positives (Negatives covered)
        
        if N_total_unique_matches == 0:
             return {"error": "Rules cover 0 samples."}

        # --- Fitness Breakdown ---
        subgroup_size_fraction = N_total_unique_matches / global_stats['total_samples']
        local_pos_fraction = N_tp / (N_total_unique_matches + eps)
        # Use the *real* global pos fraction for calculation
        gpf = global_stats['global_pos_fraction'] 
        
        quality_of_union = subgroup_size_fraction * (local_pos_fraction - gpf)
        
        overlap_ratio = total_matches_with_duplicates / (N_total_unique_matches + eps)
        overlap_ratio = max(1.0, overlap_ratio) # Clamp at 1
        
        final_fitness = quality_of_union / overlap_ratio

        chromo_stats = {
            "n_unique_matches": N_total_unique_matches,
            "n_tp": N_tp,
            "n_fp": N_fp
        }
        
        fitness_breakdown = {
            "WRAcc_Quality": quality_of_union,
            "Local_Pos_Fraction": local_pos_fraction,
            "Global_Pos_Fraction": gpf,
            "Subgroup_Size_Fraction": subgroup_size_fraction,
            "Overlap_Ratio": overlap_ratio,
            "Final_Fitness": final_fitness
        }
        
        return {
            "rule_stats": rule_stats,
            "chromo_stats": chromo_stats,
            "fitness_breakdown": fitness_breakdown
        }
        
    except Exception as e:
        return {"error": f"Error in get_chromosome_stats: {e}"}

def parse_rules_from_text(rule_text_block):
    """
    Parses a block of rule text from the summary file into a list of strings.
    This is complex because rules can span multiple lines.
    """
    rules = []
    current_rule = ""
    
    rule_start_pattern = re.compile(r"^\s*- Rule (\d+): (.*)")
    
    for line in rule_text_block:
        line = line.strip()
        match = rule_start_pattern.match(line)
        
        if match:
            # We are starting a new rule. Save the previous one if it exists.
            if current_rule:
                rules.append(current_rule.strip())
                
            # Start the new rule
            current_rule = match.group(2) # Just the rule string
        elif line.startswith("(") or "--- Rank" in line or not line:
            # This is a (Covers: ...) line or a new Rank, so the previous rule is finished.
            if current_rule:
                rules.append(current_rule.strip())
            current_rule = "" # Reset
        elif current_rule:
            # This is a continuation of a multi-line rule
            current_rule += " " + line

    # Add the last rule
    if current_rule:
        rules.append(current_rule.strip())
        
    return rules

def re_encode_rule_from_string(rule_str, gene_names_list, num_features):
    """
    Converts a rule string ("GENE=1 AND GENE2=0") back into a rule tensor.
    Returns a (mask, values) tuple, not the full rule tensor.
    """
    mask = torch.zeros(num_features, dtype=torch.int32)
    values = torch.zeros(num_features, dtype=torch.int32)
    
    gene_to_index = {name: i for i, name in enumerate(gene_names_list)}
    
    conditions = rule_str.split(" AND ")
    for cond in conditions:
        if not cond:
            continue
        try:
            gene_name, value = cond.split('=')
            val = int(value)
            if gene_name in gene_to_index:
                idx = gene_to_index[gene_name]
                mask[idx] = 1
                values[idx] = val
        except ValueError:
            print(f"Warning: Could not parse condition '{cond}'")
            continue
            
    return mask, values
