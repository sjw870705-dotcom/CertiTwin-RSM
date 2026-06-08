from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / 'results' / 'figure_data' / 'fig3_unsafe_rate_reduction_data.csv'
OUT_DIR = PROJECT_ROOT / 'figures' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / 'fig3_unsafe_rate_reduction.pdf'
OUT_PNG = OUT_DIR / 'fig3_unsafe_rate_reduction.png'


def fmt(v):
    if abs(v) < 1e-12: return '0'
    if abs(v) < 0.01: return f'{v:.4f}'
    if abs(v) < 1: return f'{v:.3f}'
    return f'{v:.2f}'


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing figure data: {INPUT}')
    df = pd.read_csv(INPUT)
    controllers = df['controller'].astype(str).tolist()
    raw = df['raw_unsafe_pct'].astype(float).to_numpy()
    shield = df['shielded_unsafe_pct'].astype(float).to_numpy()
    x = np.arange(len(controllers)); width = 0.34

    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
                         'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7.1,4.15),gridspec_kw={'height_ratios':[3.0,1.25]},sharex=True)
    ax1.bar(x-width/2, raw, width, label='Raw controller', color='#4C78A8', edgecolor='black', linewidth=0.6, zorder=3)
    ax1.bar(x+width/2, shield, width, label='CertiTwin-shielded', color='#F58518', edgecolor='black', linewidth=0.6, hatch='//', zorder=3)
    ax1.set_ylabel('Unsafe decisions (%)'); ax1.set_title('(a) Unsafe rate before and after shielding')
    ax1.set_ylim(0, max(5.0, np.nanmax(raw)*1.18)); ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
    ax1.legend(loc='upper right', ncol=2, frameon=True, edgecolor='black', columnspacing=1.2, handlelength=1.6)
    for i,v in enumerate(raw):
        ax1.text(x[i]-width/2, v+max(0.4,np.nanmax(raw)*0.015), fmt(v), ha='center', va='bottom', fontsize=7.5)
    ax2.bar(x, shield, width=0.42, color='#F58518', edgecolor='black', linewidth=0.6, hatch='//', zorder=3)
    ymax=max(0.02, np.nanmax(shield)*1.8)
    ax2.set_ylim(0,ymax); ax2.set_ylabel('Shielded\nunsafe (%)'); ax2.set_title('(b) Residual unsafe rate after shielding')
    ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
    for i,v in enumerate(shield):
        ax2.text(x[i], max(v,ymax*0.035), fmt(v), ha='center', va='bottom', fontsize=7.2)
    ax2.set_xticks(x); ax2.set_xticklabels(controllers, rotation=12, ha='right'); ax2.set_xlabel('Raw controller')
    fig.tight_layout(h_pad=0.55); fig.savefig(OUT_PDF,bbox_inches='tight'); fig.savefig(OUT_PNG,dpi=600,bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}\nSaved {OUT_PNG}')

if __name__ == '__main__': main()
