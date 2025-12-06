import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_correlation_network():
    print("Generating Regulatory Network...")
    
    # Load Data
    try:
        X = pd.read_csv("./sc_data/X_binary.csv")
        y = pd.read_csv("./sc_data/y_labels.csv")
    except:
        X = pd.read_csv("X_binary.csv")
        y = pd.read_csv("y_labels.csv")

    # Focus on Rule 2 (Iron) as the example - it has the clearest "Regulatory" feel
    # Filter for PD cells that match Rule 2 (FTH1=1)
    target_indices = (y.iloc[:,0] == 1) & (X['FTH1'] == 1)
    subgroup_data = X[target_indices]
    
    # Select key genes to build the network around
    # (Top hits from your enrichment analysis)
    key_genes = [
        'FTH1', 'FTL', 'CRYAB', 'PLP1', 'S100B', 'TMSB4X', 
        'MT-ND4', 'MT-CO3', # Contrast genes
        'LINGO1' # Contrast gene
    ]
    
    # Calculate Correlation Matrix (Spearman for binary/rank data is safer, but Pearson ok)
    # We filter for genes that actually exist in the data
    valid_genes = [g for g in key_genes if g in subgroup_data.columns]
    corr_matrix = subgroup_data[valid_genes].corr(method='pearson')
    
    # Build Graph
    G = nx.Graph()
    
    # Add Edges for strong correlations
    threshold = 0.15 # Low threshold because binary data correlations are mathematically capped
    
    for i, gene1 in enumerate(valid_genes):
        for j, gene2 in enumerate(valid_genes):
            if i >= j: continue
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(gene1, gene2, weight=corr, color='red' if corr > 0 else 'blue')

    # Plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Edges
    edges = G.edges(data=True)
    colors = [d['color'] for u, v, d in edges]
    weights = [abs(d['weight']) * 10 for u, v, d in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightgrey', edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=weights)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Signed Co-Expression Network (Rule 2: Iron/Stress State)", fontsize=15)
    plt.axis('off')
    
    plt.savefig("regulatory_network.png", dpi=300)
    print("Network saved to regulatory_network.png")

if __name__ == "__main__":
    plot_correlation_network()
