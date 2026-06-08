from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / 'results' / 'figure_data' / 'fig6_sla_threshold_robustness_data.csv'
OUT_DIR = PROJECT_ROOT / 'figures' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / 'fig6_sla_threshold_robustness.pdf'
OUT_PNG = OUT_DIR / 'fig6_sla_threshold_robustness.png'


def main():
    if not INPUT.exists(): raise FileNotFoundError(f'Missing figure data: {INPUT}')
    df=pd.read_csv(INPUT).sort_values('sla_scale')
    x=df.sla_scale.astype(float).to_numpy(); unsafe=df.max_shielded_unsafe_pct.astype(float).to_numpy(); utility=df.min_utility_retention_pct.astype(float).to_numpy()
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
                         'font.size':8.4,'axes.labelsize':8.4,'axes.titlesize':8.6,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.15,2.85),sharex=True)
    ax1.plot(x,unsafe,marker='o',linestyle='-',color='#4C78A8',linewidth=1.45,markersize=4.2,zorder=3)
    ax1.axvline(1.0,color='black',linestyle=':',linewidth=0.8,alpha=0.8)
    ax1.text(1.005, max(unsafe)*0.82 if np.nanmax(unsafe)>0 else 0.02, 'default', fontsize=7.2, va='top', ha='left')
    ax1.set_title('(a) Residual unsafe rate under SLA shifts'); ax1.set_xlabel('SLA threshold scale'); ax1.set_ylabel('Max residual unsafe (%)')
    ax1.grid(axis='both',linestyle='--',linewidth=0.5,alpha=0.55,zorder=0); ax1.xaxis.set_major_locator(MaxNLocator(nbins=5)); ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ymax=max(0.05,np.nanmax(unsafe)*1.35); ax1.set_ylim(0,ymax)
    for xi,yi in zip(x,unsafe):
        if yi>0:
            dx=0.008 if abs(xi-1.0)<1e-9 else 0.0
            ax1.text(xi+dx, yi+ymax*0.04, f'{yi:.3f}', ha='center', va='bottom', fontsize=7.1)
    ax2.plot(x,utility,marker='s',linestyle='-',color='#F58518',linewidth=1.45,markersize=4.2,zorder=3)
    ax2.axvline(1.0,color='black',linestyle=':',linewidth=0.8,alpha=0.8)
    ax2.set_title('(b) Utility retention under SLA shifts'); ax2.set_xlabel('SLA threshold scale'); ax2.set_ylabel('Min utility retention (%)')
    ax2.grid(axis='both',linestyle='--',linewidth=0.5,alpha=0.55,zorder=0); ax2.xaxis.set_major_locator(MaxNLocator(nbins=5)); ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ymin,ymax2=np.nanmin(utility),np.nanmax(utility); pad=max(0.5,(ymax2-ymin)*0.18); ax2.set_ylim(max(0,ymin-pad),ymax2+pad)
    for xi,yi in zip(x,utility):
        dx=0.008 if abs(xi-1.0)<1e-9 else 0.0
        ax2.text(xi+dx, yi+0.15, f'{yi:.2f}', ha='center', va='bottom', fontsize=7.0)
    fig.tight_layout(w_pad=1.15); fig.savefig(OUT_PDF,bbox_inches='tight'); fig.savefig(OUT_PNG,dpi=600,bbox_inches='tight'); plt.close(fig)
    print(f'Saved {OUT_PDF}\nSaved {OUT_PNG}')
if __name__=='__main__': main()
