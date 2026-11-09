"""Export aligned training-progress figures and a late-stage validation detail."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/real_lift_20261107'

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, (key, label) in enumerate([('finetune', 'Fine-tuning (source normalization)'), ('scratch', 'From scratch (real train-only normalization)')]):
        run = ROOT / 'outputs' / ('lift_' + key + '_20261107_run01')
        logs = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines()]
        vals = [x for x in logs if 'ema_val_loss' in x]
        axes[0,col].plot([x['equivalent_epochs'] for x in logs], [x['train_loss'] for x in logs], lw=.4)
        axes[1,col].plot([x['equivalent_epochs'] for x in vals], [x['ema_val_loss'] for x in vals], lw=1)
        axes[0,col].set_title(label)
        for ax in axes[:,col]:
            ax.set_yscale('log'); ax.set_xlim(0,500); ax.grid(alpha=.2)
        axes[1,col].set_xlabel('Equivalent training epochs')
        if key == 'scratch':
            detail, ax = plt.subplots(figsize=(10,4.5))
            late = [x for x in vals if x['equivalent_epochs'] >= 20]
            ax.plot([x['equivalent_epochs'] for x in late], [x['ema_val_loss'] for x in late])
            best = json.loads((run/'analysis/summary.json').read_text())['best_saved_checkpoint']
            ax.axvline(best['equivalent_epochs'], color='tab:orange', ls='--', label='Best saved checkpoint: epoch 345')
            ax.set(xlabel='Training epochs', ylabel='Full-validation EMA denoising loss', title='From scratch: validation detail from epoch 20 (unsmoothed)')
            ax.grid(alpha=.2); ax.legend(); detail.tight_layout()
            for ext in ['png','pdf']: detail.savefig(OUT/('scratch_validation_detail.'+ext), dpi=180)
            plt.close(detail)
    axes[0,0].set_ylabel('Raw training denoising loss (log)')
    axes[1,0].set_ylabel('Full-validation EMA denoising loss (log)')
    fig.suptitle('Real Lift DP: aligned training progress; raw curves, no smoothing')
    fig.text(.5,.015,'Different normalization, learning rates and budgets: loss magnitudes do not establish relative robot success.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ['png','pdf']: fig.savefig(OUT/('training_progress_comparison.'+ext),dpi=180)
    plt.close(fig)

if __name__ == '__main__': main()
