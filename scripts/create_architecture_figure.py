from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, text):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.035", linewidth=1.2, facecolor="white", edgecolor="black")
    ax.add_patch(p); ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=9, wrap=True)

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a,b, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="black"))


def main(out="outputs_astar/publication_figures_astar/dare_copdnet_architecture"):
    fig, ax = plt.subplots(figsize=(10,5.6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    box(ax,(0.04,0.43),0.14,0.13,"Respiratory\naudio cycle")
    box(ax,(0.24,0.72),0.18,0.12,"Log-mel\nspectrogram")
    box(ax,(0.24,0.43),0.18,0.12,"Temporal\nspectrogram")
    box(ax,(0.24,0.14),0.18,0.12,"Handcrafted\nacoustic features")
    box(ax,(0.50,0.72),0.18,0.12,"EfficientNet +\nMixStyle")
    box(ax,(0.50,0.43),0.18,0.12,"CRNN + temporal\nattention")
    box(ax,(0.50,0.14),0.18,0.12,"Feature MLP")
    box(ax,(0.73,0.43),0.14,0.13,"Adaptive\ntri-view gate")
    box(ax,(0.73,0.19),0.14,0.11,"Domain head\nGRL + CORAL/IRM")
    box(ax,(0.90,0.58),0.08,0.12,"Binary\nCOPD")
    box(ax,(0.90,0.36),0.08,0.12,"Multiclass\ndiagnosis")
    for y in [0.785,0.495,0.205]: arrow(ax,(0.18,0.495),(0.24,y))
    arrow(ax,(0.42,0.78),(0.50,0.78)); arrow(ax,(0.42,0.49),(0.50,0.49)); arrow(ax,(0.42,0.20),(0.50,0.20))
    arrow(ax,(0.68,0.78),(0.73,0.52)); arrow(ax,(0.68,0.49),(0.73,0.50)); arrow(ax,(0.68,0.20),(0.73,0.47))
    arrow(ax,(0.80,0.43),(0.94,0.58)); arrow(ax,(0.80,0.43),(0.94,0.48)); arrow(ax,(0.80,0.43),(0.80,0.30))
    ax.text(0.5,0.94,"DARE-COPDNet: domain-robust binary and multiclass respiratory-sound classification",ha="center",fontsize=12,weight="bold")
    out=Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight'); fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')

if __name__ == '__main__': main()
