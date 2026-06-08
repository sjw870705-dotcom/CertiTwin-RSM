from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / 'results' / 'figure_data' / 'fig5_certificate_inflation_data.csv'
OUT_DIR = PROJECT_ROOT / 'figures' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / 'fig5_certificate_inflation.pdf'
OUT_PNG = OUT_DIR / 'fig5_certificate_inflation.png'


def main():
    if not INPUT.exists(): raise FileNotFoundError(f'Missing figure data: {INPUT}')
    df=pd.read_csv(INPUT).sort_values(['noise_scale','alpha'])
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
                         'font.size':8.4,'axes.labelsize':8.4,'axes.titlesize':8.6,'legend.fontsize':7.2,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    markers=['o','s','^','D','v']; linestyles=['-','--','-.',':']; colors=['#4C78A8','#F58518','#54A24B','#E45756']
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.15,3.0),sharex=True)
    for i,noise in enumerate(sorted(df.noise_scale.unique())):
        g=df[df.noise_scale==noise].sort_values('alpha'); label=f'noise={noise:.2f}'
        ax1.plot(g.alpha,g.max_shielded_unsafe_pct,marker=markers[i%len(markers)],linestyle=linestyles[i%len(linestyles)],color=colors[i%len(colors)],linewidth=1.35,markersize=4.0,label=label,zorder=3)
        ax2.plot(g.alpha,g.min_utility_retention_pct,marker=markers[i%len(markers)],linestyle=linestyles[i%len(linestyles)],color=colors[i%len(colors)],linewidth=1.35,markersize=4.0,label=label,zorder=3)
    ax1.set_title('(a) Residual unsafe rate under perturbed twin'); ax1.set_xlabel('Certificate inflation factor $\\alpha$'); ax1.set_ylabel('Max residual unsafe (%)')
    ax2.set_title('(b) Utility-retention tradeoff'); ax2.set_xlabel('Certificate inflation factor $\\alpha$'); ax2.set_ylabel('Min utility retention (%)')
    for ax in (ax1,ax2):
        ax.grid(axis='both',linestyle='--',linewidth=0.5,alpha=0.55,zorder=0); ax.xaxis.set_major_locator(MaxNLocator(nbins=5)); ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    h,l=ax1.get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',ncol=min(len(l),4),frameon=True,edgecolor='black',bbox_to_anchor=(0.5,1.015),columnspacing=0.85,handlelength=1.8,handletextpad=0.45,borderpad=0.35)
    fig.tight_layout(rect=[0,0,1,0.91],w_pad=1.15); fig.savefig(OUT_PDF,bbox_inches='tight'); fig.savefig(OUT_PNG,dpi=600,bbox_inches='tight'); plt.close(fig)
    print(f'Saved {OUT_PDF}\nSaved {OUT_PNG}')
if __name__=='__main__': main()
