import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_detailed_architecture():
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # --- UTILS ---
    def add_box(x, y, w, h, color, label, alpha=1.0, ec='black', text_kw={}):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=ec, facecolor=color, alpha=alpha, zorder=10)
        ax.add_patch(rect)
        if label:
            cx, cy = x + w/2, y + h/2
            ax.text(cx, cy, label, ha='center', va='center', **text_kw)
        return rect

    # --- 1. CLUSTER LEVEL (Top) ---
    # Background
    add_box(0.05, 0.82, 0.9, 0.15, '#f0f0f0', "")
    ax.text(0.06, 0.95, "TIER 2: Distributed Parallelism (100 SLURM Jobs)", fontsize=14, fontweight='bold', ha='left')

    # Nodes
    for i in range(5):
        x_start = 0.1 + i*0.16
        col = '#90EE90' if i == 0 else 'white' # Highlight Job 0
        label = f"Job {i}" if i < 4 else "Job 99"
        add_box(x_start, 0.84, 0.12, 0.1, col, label + "\n(PVC GPU)")
        if i == 3: 
            ax.text(x_start + 0.06, 0.89, "...", fontsize=20, ha='center')
            
    # Arrow down
    ax.annotate("", xy=(0.16, 0.78), xytext=(0.16, 0.84), arrowprops=dict(arrowstyle="->", lw=2))

    # --- 2. TENSOR LEVEL (Middle) ---
    # Background container
    add_box(0.05, 0.45, 0.9, 0.33, '#e6f3ff', "")
    ax.text(0.06, 0.76, "TIER 1: Vectorized Intra-Node Parallelism (Single Job)", fontsize=14, fontweight='bold', ha='left')

    # --- POPULATION TENSOR (Left) ---
    # Draw a "stack" to show the 10 permutations
    # The front face is the 55 x 12008 matrix
    
    # Shadow blocks for permutations 1-9
    for i in range(3):
        add_box(0.12 + i*0.01, 0.51 + i*0.01, 0.2, 0.2, '#4a90e2', "", alpha=0.3)
    
    # Main Block (Permutation 0)
    pop_x, pop_y = 0.1, 0.49
    pop_w, pop_h = 0.2, 0.2
    add_box(pop_x, pop_y, pop_w, pop_h, '#4a90e2', "")
    
    # Labels for Pop
    ax.text(pop_x - 0.03, pop_y + pop_h/2, "Population:\n55 Individuals", rotation=90, ha='center', va='center', fontweight='bold')
    ax.text(pop_x + pop_w/2, pop_y + pop_h + 0.02, "Chromosome Width:\n12,008 bits (int8)", ha='center', va='bottom', fontsize=9)
    ax.text(pop_x + 0.25, pop_y + 0.25, "Batch Dim:\n10 Permutations", fontsize=10, color='grey')

    # --- DATA TENSOR (Right) ---
    # Shadow blocks
    for i in range(3):
        add_box(0.62 + i*0.01, 0.51 + i*0.01, 0.2, 0.2, '#ff9f43', "", alpha=0.3)
    
    # Main Block
    data_x, data_y = 0.6, 0.49
    data_w, data_h = 0.2, 0.2
    add_box(data_x, data_y, data_w, data_h, '#ff9f43', "Data Matrix\n(ReadOnly)")
    
    # Labels for Data
    ax.text(data_x + data_w + 0.03, data_y + data_h/2, "Cells:\n6,385", rotation=270, ha='center', va='center', fontweight='bold')
    ax.text(data_x + data_w/2, data_y - 0.02, "Features: 1,001", ha='center', va='top', fontsize=9)

    # BROADCAST OPERATION
    ax.text(0.45, 0.59, "BROADCAST\nEVALUATION", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.annotate("", xy=(0.58, 0.59), xytext=(0.32, 0.59), arrowprops=dict(arrowstyle="simple", fc="black"))
    ax.text(0.45, 0.52, "10 x 55 x 6385\nOperations/Clock", ha='center', fontsize=9, color='#d63031')

    # --- 3. CHROMOSOME ARCHITECTURE (Bottom - The "Zoom In") ---
    # Background
    add_box(0.05, 0.02, 0.9, 0.38, '#fff5e6', "")
    ax.text(0.06, 0.38, "CHROMOSOME ARCHITECTURE (One Individual)", fontsize=14, fontweight='bold', ha='left')

    # Connection Line (Zoom)
    # Draw a highlight row in the population matrix
    add_box(pop_x, pop_y + 0.15, pop_w, 0.02, 'yellow', "", alpha=0.5, ec='none')
    # Arrow down
    ax.annotate("", xy=(0.5, 0.35), xytext=(pop_x + pop_w/2, pop_y + 0.15), 
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="dashed", color="black"))

    # The Chromosome Bar
    chrom_x, chrom_y = 0.1, 0.22
    total_w = 0.8
    rule_w = (total_w - 0.05) / 6
    
    # Header (Rule Count)
    add_box(chrom_x, chrom_y, 0.05, 0.08, '#a55eea', "Header\n(8 bits)", text_kw={'color':'white', 'fontsize':8})
    
    # Rules 1-6
    colors = ['#26de81', '#2bcbba'] * 3
    for i in range(6):
        rx = chrom_x + 0.05 + i*rule_w
        label = f"Rule {i+1}"
        add_box(rx, chrom_y, rule_w, 0.08, colors[i], label)
    
    ax.text(0.5, 0.31, "12,008 bits (int8 packed)", ha='center', fontweight='bold')

    # --- RULE BREAKDOWN (Zoom into Rule 1) ---
    # Zoom lines
    ax.plot([chrom_x + 0.05, 0.15], [chrom_y, 0.15], 'k--', lw=1)
    ax.plot([chrom_x + 0.05 + rule_w, 0.85], [chrom_y, 0.15], 'k--', lw=1)

    # The Rule Detail
    rule_detail_y = 0.05
    add_box(0.15, rule_detail_y, 0.35, 0.1, '#26de81', "GENE MASK (1000 bits)\n(Which genes matter?)")
    add_box(0.50, rule_detail_y, 0.35, 0.1, '#fed330', "GENE VALUES (1000 bits)\n(1 = High, 0 = Low)")

    # Logic Annotation
    ax.text(0.5, 0.02, "Rule Logic: (Data == Value) OR (Mask == 0)", ha='center', fontsize=12,  bbox=dict(facecolor='white', alpha=1.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("tensor_architecture_v2.png", dpi=300)
    print("Saved tensor_architecture_v2.png")

draw_detailed_architecture()
