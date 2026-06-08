from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / 'results' / 'figure_data' / 'fig4_action_distance_ablation_data.csv'
OUT_DIR = PROJECT_ROOT / 'figures' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / 'fig4_action_distance_ablation.pdf'
OUT_PNG = OUT_DIR / 'fig4_action_distance_ablation.png'

ORDER=['Full-CertiTwin','Raw-Pred-Safe','Uncalibrated-Upper','No-Distance-Projection','Risk-Min-Projection']

def main():
    if not INPUT.exists(): raise FileNotFoundError(f'Missing figure data: {INPUT}')
    df=pd.read_csv(INPUT)
    df['variant']=df['variant'].astype(str)
    df['_order']=df['variant'].apply(lambda x: ORDER.index(x) if x in ORDER else 999)
    df=df.sort_values(['_order','variant']).drop(columns=['_order'])
    variants=df['variant'].tolist(); dist=df['mean_action_distance'].astype(float).to_numpy(); y=np.arange(len(variants))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
                         'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.8,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.1,3.05))
    colors=[]; hatches=[]
    for v in variants:
        if v=='Full-CertiTwin': colors.append('#4C78A8'); hatches.append('')
        elif v in ['No-Distance-Projection','Risk-Min-Projection']: colors.append('#F58518'); hatches.append('//')
        else: colors.append('#B0B0B0'); hatches.append('..')
    bars=ax.barh(y,dist,color=colors,edgecolor='black',linewidth=0.6,zorder=3)
    for b,h in zip(bars,hatches): b.set_hatch(h)
    ax.set_yticks(y); ax.set_yticklabels(variants); ax.invert_yaxis()
    ax.set_xlabel('Mean action distance'); ax.set_title('Action-distance comparison under ablation variants')
    ax.grid(axis='x',linestyle='--',linewidth=0.5,alpha=0.6,zorder=0); ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    xmax=max(1.0,np.nanmax(dist)*1.30); ax.set_xlim(0,xmax)
    for yi,v in enumerate(dist): ax.text(v+xmax*0.015,yi,f'{v:.2f}',va='center',ha='left',fontsize=8)
    if 'Full-CertiTwin' in variants:
        full=float(df.loc[df['variant']=='Full-CertiTwin','mean_action_distance'].iloc[0])
        ax.axvline(full,color='black',linestyle=':',linewidth=0.8,alpha=0.75,zorder=2)
        ax.text(full+xmax*0.01,-0.42,'Full-CertiTwin reference',ha='left',va='center',fontsize=7.2)
    fig.tight_layout(); fig.savefig(OUT_PDF,bbox_inches='tight'); fig.savefig(OUT_PNG,dpi=600,bbox_inches='tight'); plt.close(fig)
    print(f'Saved {OUT_PDF}\nSaved {OUT_PNG}')
if __name__=='__main__': main()
