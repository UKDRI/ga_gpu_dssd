import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_tensor_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    color_pop = '#1f77b4'  # Blue
    color_data = '#ff7f0e' # Orange
    color_slurm = '#2ca02c' # Green
    
    # --- PART 1: SLURM Hierarchy (Top) ---
    
    # Draw "Cluster" box
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.65), 0.9, 0.3, boxstyle="round,pad=0.02", 
                                      edgecolor='black', facecolor='#f0f0f0', zorder=0))
    ax.text(0.55, 0.92, "DAWN HPC Cluster (100 Nodes)", fontsize=14, fontweight='bold')

    # Draw Jobs
    for i, x_pos in enumerate([0.6, 0.8, 1.0, 1.2]):
        label = f"Job {i}" if i < 3 else "Job 99"
        if i == 3: x_pos += 0.1 # Spacing for dots
        
        # Job Box
        ax.add_patch(patches.Rectangle((x_pos, 0.7), 0.15, 0.15, edgecolor='black', facecolor='white'))
        ax.text(x_pos + 0.075, 0.775, label, ha='center', va='center', fontsize=10)
        
        # GPU inside Job
        ax.add_patch(patches.Rectangle((x_pos + 0.025, 0.72), 0.1, 0.04, edgecolor='black', facecolor=color_slurm, alpha=0.3))
        ax.text(x_pos + 0.075, 0.74, "PVC GPU", ha='center', va='center', fontsize=7)

        # Arrow down to detailed view
        if i == 0:
            ax.annotate("", xy=(0.35, 0.55), xytext=(x_pos + 0.075, 0.7),
                        arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(1.15, 0.775, "...", fontsize=20)

    # --- PART 2: Tensor Architecture (Bottom - Detailed View of ONE Job) ---
    
    # GPU Memory Box
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.05), 0.6, 0.5, boxstyle="round,pad=0.02", 
                                      edgecolor='black', facecolor='#e8f4f8', zorder=0))
    ax.text(0.1, 0.52, "Single GPU Memory (Job Task)", fontsize=14, fontweight='bold')

    # 3D Tensor Visualization Helper
    def draw_3d_box(ax, origin, size, color, label, dims):
        x, y = origin
        w, h, d = size
        
        # Front face
        ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='black', facecolor=color, alpha=0.8, zorder=2))
        # Top face
        ax.add_patch(patches.Polygon([[x, y+h], [x+d, y+h+d], [x+w+d, y+h+d], [x+w, y+h]], 
                                     edgecolor='black', facecolor=color, alpha=0.6, zorder=1))
        # Side face
        ax.add_patch(patches.Polygon([[x+w, y], [x+w+d, y+d], [x+w+d, y+h+d], [x+w, y+h]], 
                                     edgecolor='black', facecolor=color, alpha=0.4, zorder=1))
        
        # Labels
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=3)
        
        # Dimensions
        ax.text(x + w/2, y - 0.03, dims[1], ha='center', va='top', fontsize=9) # Width dimension
        ax.text(x - 0.02, y + h/2, dims[0], ha='right', va='center', fontsize=9, rotation=90) # Height dimension
        ax.text(x + w + d + 0.01, y + d/2, dims[2], ha='left', va='center', fontsize=9) # Depth dimension

    # Population Tensor
    draw_3d_box(ax, (0.1, 0.25), (0.15, 0.2, 0.05), color_pop, "Population\nTensor", 
                ("10 Permutations", "12,008 Bits\n(Rules)", "55 Indiv."))

    # Data Tensor
    draw_3d_box(ax, (0.35, 0.25), (0.15, 0.2, 0.05), color_data, "Data\nTensor", 
                ("10 Permutations", "1,001 Cols\n(Genes)", "6,385 Cells"))

    # Operations
    ax.text(0.3, 0.35, "BROADCAST\nEVALUATION", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate("", xy=(0.34, 0.35), xytext=(0.26, 0.35), arrowprops=dict(arrowstyle="simple", fc="black"))

    # Description Text
    desc = (
        "Tier 1 Parallelism (Intra-GPU):\n"
        "• 10 Independent GA populations run simultaneously\n"
        "• Bit-packed int8 tensors for max throughput\n"
        "• Zero-copy broadcasting for fitness eval"
    )
    ax.text(0.1, 0.1, desc, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    desc2 = (
        "Tier 2 Parallelism (Inter-Node):\n"
        "• 100 SLURM Array Jobs\n"
        "• Total Scale: 100 Jobs × 10 Permutations\n"
        "   = 1,000 Concurrent Searches"
    )
    ax.text(0.8, 0.55, desc2, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.0)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("tensor_architecture.png", dpi=300)
    print("Saved tensor_architecture.png")

draw_tensor_architecture()
