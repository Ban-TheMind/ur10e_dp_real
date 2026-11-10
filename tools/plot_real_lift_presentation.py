"""PPT figures for the real-normalized 100-epoch fine-tune and 500-epoch scratch run."""
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentations/real_lift_training/output/zzj_11.09_figures'
RUNS = [('finetune', '微调', ROOT/'outputs/lift_finetune_real_norm_100ep_run01', '#2E75B6'),
        ('scratch', '从零训练', ROOT/'outputs/lift_scratch_20261107_run01', '#AE5A24')]
INK, MUTED, GRID = '#000000', '#000000', '#E6E6E6'
# Register the installed fonts explicitly: Matplotlib's cache may predate YaHei.
from matplotlib import font_manager
for family in ('Microsoft YaHei', 'Times New Roman'):
    for pattern in (family, family+':style=Bold'):
        font_path=subprocess.check_output(['fc-match','-f','%{file}',pattern],text=True).strip()
        font_manager.fontManager.addfont(font_path)
plt.rcParams.update({
 'font.family':['Times New Roman','Microsoft YaHei'], 'font.size':18,
 'axes.unicode_minus':False, 'pdf.fonttype':42, 'ps.fonttype':42,
 'svg.fonttype':'none', 'axes.linewidth':0.8, 'axes.edgecolor':'#BCC6D1',
 'axes.labelcolor':MUTED, 'xtick.color':MUTED, 'ytick.color':MUTED,
 'xtick.labelsize':16, 'ytick.labelsize':16, 'axes.labelsize':18,
 'axes.titlesize':21, 'figure.facecolor':'white', 'savefig.facecolor':'white',
 'lines.solid_capstyle':'round', 'axes.axisbelow':True,
})

def load_runs():
    data=[]
    for key,label,path,color in RUNS:
        meta=json.loads((path/'run.json').read_text())
        completed=json.loads((path/'completion.json').read_text())
        logs=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]
        assert len(logs)==completed['updates']==meta['settings']['max_updates']
        assert [x['updates'] for x in logs]==list(range(1,len(logs)+1))
        assert all(np.isfinite(x['train_loss']) and ('ema_val_loss' not in x or np.isfinite(x['ema_val_loss'])) for x in logs)
        assert meta['normalizer']=='real_train_only' and meta['val_windows']==1184
        assert meta['settings']['validation_batches']==0
        vals=[x for x in logs if 'ema_val_loss' in x]
        best=min(vals,key=lambda x:x['ema_val_loss'])
        saved=json.loads((path/'best_val.json').read_text())
        assert best['updates']==saved['updates'] and best['ema_val_loss']==saved['ema_val_loss']
        epochs=np.array([x['equivalent_epochs'] for x in logs])
        batch_weights=np.rint(np.diff(np.r_[0,epochs])*meta['train_windows']).astype(int)
        epoch_index=np.ceil(epochs-1e-8).astype(int)
        aggregated=[]
        for e in np.unique(epoch_index):
            mask=epoch_index==e
            aggregated.append(dict(epoch=int(e),train_loss=float(np.average(np.array([x['train_loss'] for x in logs])[mask],weights=batch_weights[mask])),windows=int(batch_weights[mask].sum())))
        assert all(x['windows']==meta['train_windows'] for x in aggregated)
        data.append(dict(key=key,label=label,path=path,color=color,meta=meta,completed=completed,logs=logs,vals=vals,best=best,aggregated=aggregated))
    a,b=data
    assert a['meta']['split']==b['meta']['split']
    assert a['meta']['dataset_hashes']==b['meta']['dataset_hashes']
    assert a['meta']['policy_config']==b['meta']['policy_config']
    assert a['meta']['settings']['seed']==b['meta']['settings']['seed']==42
    # Check actual EMA normalizer parameters, not just the manifest's label.
    import torch
    states=[torch.load(d['path']/'best_val.pt',map_location='cpu',weights_only=False) for d in data]
    keys=[k for k in states[0]['ema_model'] if k.startswith('normalizer.')]
    assert keys and keys==[k for k in states[1]['ema_model'] if k.startswith('normalizer.')]
    assert all(torch.equal(states[0]['ema_model'][k],states[1]['ema_model'][k]) for k in keys)
    for d,state in zip(data,states):assert state['updates']==d['best']['updates']
    return data,len(keys)

plt.rcParams.update({'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black','axes.labelsize':21,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':19})

def base():
    fig,ax=plt.subplots(figsize=(12.8,5.65))
    fig.subplots_adjust(left=.125,right=.97,bottom=.21,top=.94)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#7B7B7B')
    ax.grid(axis='y',color='#E6E6E6',linewidth=.8)
    ax.set_axisbelow(True);ax.set_yscale('log');ax.set_ylim(.014,.15)
    ax.yaxis.set_major_locator(FixedLocator([.015,.025,.05,.1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter());ax.tick_params(which='minor',length=0)
    ax.tick_params(which='major',pad=8)
    ax.set_xlabel('训练 Epoch',labelpad=12)
    ax.set_ylabel('EMA 验证去噪损失（对数）',labelpad=14)
    return fig,ax

def export(fig,name):
    for t in fig.findobj(Text):
        t.set_color('#000000')
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/(name+'.'+ext),dpi=300)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    data,n=load_runs()
    # Training loss: raw model, sample-weighted mean within each full epoch.
    fig,ax=base()
    ax.set_ylabel('训练去噪损失（对数）',labelpad=14)
    ax.set_ylim(.008,1.2);ax.set_yticks([.01,.03,.1,.3,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    for d in data:
        a=d['aggregated']
        ax.plot([v['epoch'] for v in a],[v['train_loss'] for v in a],color=d['color'],lw=2.4,label=f"{d['label']}（{int(d['completed']['equivalent_epochs'])} epoch）")
    ax.set_xlim(0,500);ax.set_xticks([0,100,200,300,400,500]);ax.legend(frameon=False,loc='upper right',labelcolor='black')
    export(fig,'zzj_11.09_training_denoising')
    # Validation loss: identical complete held-out split and fixed evaluation noise.
    fig,ax=base()
    for d in data:
        v=d['vals'];best=d['best']
        ax.plot([x['equivalent_epochs'] for x in v],[x['ema_val_loss'] for x in v],color=d['color'],lw=2.6,label=f"{d['label']}（{int(d['completed']['equivalent_epochs'])} epoch）")
        ax.scatter(best['equivalent_epochs'],best['ema_val_loss'],color=d['color'],s=65,edgecolor='white',linewidth=1.5,zorder=4)
        xytext=(135,.0185) if d['key']=='finetune' else (365,.040)
        ax.annotate(f"{d['label']}最佳：epoch {best['equivalent_epochs']:.0f}\n{best['ema_val_loss']:.5f}",xy=(best['equivalent_epochs'],best['ema_val_loss']),xytext=xytext,fontsize=18,ha='left',va='center',color='black',arrowprops={'arrowstyle':'-','color':'#777777','lw':1.0})
    ax.set_xlim(0,500);ax.set_xticks([0,100,200,300,400,500]);ax.legend(frameon=False,loc='upper right',labelcolor='black')
    export(fig,'zzj_11.09_validation_ema')
    fig,ax=base()
    for d in data:
        v=[x for x in d['vals'] if x['equivalent_epochs']<=100]
        ax.plot([x['equivalent_epochs'] for x in v],[x['ema_val_loss'] for x in v],color=d['color'],lw=2.6,marker='o',markersize=4,label=d['label'])
        ax.text(103,v[-1]['ema_val_loss'],f"{v[-1]['ema_val_loss']:.5f}",fontsize=18,ha='left',va='center',color='black')
    ax.set_xlim(0,118);ax.set_xticks([0,20,40,60,80,100]);ax.legend(frameon=False,loc='upper right',labelcolor='black')
    export(fig,'zzj_11.09_validation_100epoch')
    vals=[dict(method=d['key'],epoch=v['equivalent_epochs'],updates=v['updates'],ema_validation_loss=v['ema_val_loss']) for d in data for v in d['vals']]
    agg=[dict(method=d['key'],**v) for d in data for v in d['aggregated']]
    for name,rows in [('validation_data.csv',vals),('epoch_training_means.csv',agg)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'normalizer_tensor_count':n,'normalizers_exactly_equal':True,'training_loss':'raw-model sample-weighted epoch mean','validation_loss':'EMA full 1184 windows every 5 epochs fixed seed 43','runs':{d['key']:{'source':str(d['path']),'epochs':d['completed']['equivalent_epochs'],'updates':d['completed']['updates'],'best_epoch':d['best']['equivalent_epochs'],'best_validation_loss':d['best']['ema_val_loss'],'final_epoch_training_mean':d['aggregated'][-1]['train_loss'],'epoch_100_validation_loss':next(v['ema_val_loss'] for v in d['vals'] if v['equivalent_epochs']==100)} for d in data}}
    (OUT/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'README.md').write_text("# zzj_11.09 figures\n\nPPT page 5 uses training_denoising: raw-model sample-weighted epoch means. Page 6 uses validation_ema: full 1184-window EMA validation every five epochs with seed 43. validation_100epoch is a standalone supplementary figure. All text is black; Chinese Microsoft YaHei, Latin/digits Times New Roman. Each figure has PNG/PDF/SVG. Validation is unsmoothed. Fine-tuning ends at epoch100, never extrapolated to 500. Same data split, hashes, architecture and all36 normalization tensors; different learning rates/schedules, one seed. No robot success rate measured.\n\nReproduce: OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv_train/bin/python tools/plot_real_lift_presentation.py\nThe variables limit CPU threads; the project interpreter regenerates figures without starting training.\n")
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
