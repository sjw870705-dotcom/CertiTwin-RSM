from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / 'results' / 'figure_data' / 'figC1_runtime_scaling_data.csv'
FIT_INPUT = PROJECT_ROOT / 'results' / 'paper_tables' / 'table_CIII_runtime_linear_fit.csv'
OUT_DIR = PROJECT_ROOT / 'figures' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / 'figC1_runtime_scaling.pdf'
OUT_PNG = OUT_DIR / 'figC1_runtime_scaling.png'


def fit_from_data(df):
    x=df.K.astype(float).to_numpy(); y=df.mean_latency_ms.astype(float).to_numpy()
    slope, intercept=np.polyfit(x,y,1); pred=slope*x+intercept
    r2=1-np.sum((y-pred)**2)/(np.sum((y-np.mean(y))**2)+1e-12)
    return float(slope), float(intercept), float(r2)


def main():
    if not INPUT.exists(): raise FileNotFoundError(f'Missing figure data: {INPUT}')
    df=pd.read_csv(INPUT).sort_values('K')
    if FIT_INPUT.exists():
        fit=pd.read_csv(FIT_INPUT).iloc[0]
        slope=float(fit['slope_ms_per_candidate']); intercept=float(fit['intercept_ms']); r2=float(fit['r2'])
    else:
        slope,intercept,r2=fit_from_data(df)
    x=df.K.astype(float).to_numpy(); mean=df.mean_latency_ms.astype(float).to_numpy(); p99=df.p99_latency_ms.astype(float).to_numpy()
    xfit=np.linspace(np.min(x),np.max(x),300); yfit=slope*xfit+intercept
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
                         'font.size':8.4,'axes.labelsize':8.4,'axes.titlesize':8.6,'legend.fontsize':7.4,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.15,2.9),sharex=True)
    ax1.plot(x,mean,marker='o',linestyle='-',color='#4C78A8',linewidth=1.55,markersize=4.2,label='Mean latency',zorder=3)
    ax1.plot(xfit,yfit,linestyle=':',color='black',linewidth=1.25,label='Linear fit',zorder=2)
    ax1.set_title('(a) Mean latency scaling'); ax1.set_xlabel('Synthetic candidate size $K$'); ax1.set_ylabel('Mean latency (ms)')
    ax1.grid(axis='both',linestyle='--',linewidth=0.5,alpha=0.55,zorder=0); ax1.xaxis.set_major_locator(MaxNLocator(nbins=5)); ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ymin=np.nanmin(np.concatenate([mean,yfit])); ymax=np.nanmax(np.concatenate([mean,yfit])); pad=max(0.0015,(ymax-ymin)*0.22); ax1.set_ylim(ymin-pad,ymax+pad)
    ax1.text(0.05,0.92,f'$R^2={r2:.4f}$\n' + f'slope={slope:.2e} ms/candidate', transform=ax1.transAxes, ha='left', va='top', fontsize=7.2, bbox=dict(boxstyle='round,pad=0.22',facecolor='white',edgecolor='0.55',alpha=0.92))
    ax1.legend(loc='lower right', frameon=True, edgecolor='black', handlelength=1.8, borderpad=0.35)
    ax1.text(x[0],mean[0]+pad*0.35,f'{mean[0]:.4f}',ha='center',va='bottom',fontsize=6.8)
    ax1.text(x[-1],mean[-1]+pad*0.35,f'{mean[-1]:.4f}',ha='center',va='bottom',fontsize=6.8)
    ax2.plot(x,p99,marker='s',linestyle='--',color='#F58518',linewidth=1.5,markersize=4.0,label='P99 latency',zorder=3)
    ax2.set_title('(b) Tail latency'); ax2.set_xlabel('Synthetic candidate size $K$'); ax2.set_ylabel('P99 latency (ms)')
    ax2.grid(axis='both',linestyle='--',linewidth=0.5,alpha=0.55,zorder=0); ax2.xaxis.set_major_locator(MaxNLocator(nbins=5)); ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    pmin,pmax=np.nanmin(p99),np.nanmax(p99); ppad=max(0.006,(pmax-pmin)*0.18); ax2.set_ylim(pmin-ppad,pmax+ppad)
    ax2.legend(loc='upper left', frameon=True, edgecolor='black', handlelength=1.8, borderpad=0.35)
    ax2.text(x[-1],p99[-1]+ppad*0.25,f'{p99[-1]:.4f}',ha='center',va='bottom',fontsize=6.8)
    fig.tight_layout(w_pad=1.15); fig.savefig(OUT_PDF,bbox_inches='tight'); fig.savefig(OUT_PNG,dpi=600,bbox_inches='tight'); plt.close(fig)
    print(f'Saved {OUT_PDF}\nSaved {OUT_PNG}')
if __name__=='__main__': main()
