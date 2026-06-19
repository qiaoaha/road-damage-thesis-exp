#!/usr/bin/env python3
import csv, json, re, os
from pathlib import Path
BASE = Path('/root/autodl-tmp/road_damage_exp')
RUN_ROOT = BASE / 'runs_ch4_base80_wandb'
OUT_CSV = RUN_ROOT / 'ch4_all_yolos_results.csv'
OUT_MD = RUN_ROOT / 'ch4_all_yolos_results.md'
FAMILIES = ['yolov5','yolov8','yolov11']
DATASETS = ['base80_only','base80_plus_random_200','base80_plus_lpips_200','base80_plus_ours_200']

def read_status():
    d = {}
    p = RUN_ROOT/'ch4_training_status.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                j=json.loads(line); key=(j.get('model_family'), j.get('dataset_variant')); d[key]=j
            except Exception: pass
    return d

def parse_results_csv(path):
    if not path.exists(): return {}
    try:
        rows=list(csv.DictReader(path.open(newline='', encoding='utf-8', errors='replace')))
        if not rows: return {}
        row=rows[-1]
        def pick(*names):
            for n in names:
                for k,v in row.items():
                    if k and k.strip()==n and v not in (None,''):
                        try: return float(v)
                        except Exception: return v
            return ''
        return {
            'train_precision': pick('metrics/precision(B)','metrics/precision'),
            'train_recall': pick('metrics/recall(B)','metrics/recall'),
            'train_mAP50': pick('metrics/mAP50(B)','metrics/mAP_0.5'),
            'train_mAP50_95': pick('metrics/mAP50-95(B)','metrics/mAP_0.5:0.95'),
        }
    except Exception as e:
        return {'parse_error':repr(e)}

def find_test_metrics(test_dir):
    metrics = {}
    candidates = list(test_dir.glob('**/results.csv')) + list(test_dir.glob('**/*.json'))
    for p in candidates:
        if p.name == 'results.csv': metrics.update(parse_results_csv(p))
    return metrics

def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    status=read_status(); rows=[]
    for fam in FAMILIES:
        for ds in DATASETS:
            run=f'{fam}_{ds}'
            train_dir=RUN_ROOT/fam/run
            test_dir=RUN_ROOT/f'{fam}_test'/run
            best=train_dir/'weights/best.pt'
            r={
                'model_family':fam,'dataset_variant':ds,'run_name':run,
                'status':status.get((fam,ds),{}).get('status','missing'),
                'batch':status.get((fam,ds),{}).get('batch',''),
                'best_pt_path':str(best) if best.exists() else '',
                'results_csv_path':str(train_dir/'results.csv') if (train_dir/'results.csv').exists() else '',
                'test_dir':str(test_dir) if test_dir.exists() else '',
            }
            r.update(parse_results_csv(train_dir/'results.csv'))
            rows.append(r)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines=['# Chapter 4 YOLO all-model results','',f'Output root: `{RUN_ROOT}`','']
    lines.append('|model|dataset|status|batch|best.pt|mAP50|mAP50-95|')
    lines.append('|---|---|---:|---:|---|---:|---:|')
    for r in rows:
        lines.append(f"|{r['model_family']}|{r['dataset_variant']}|{r.get('status','')}|{r.get('batch','')}|{bool(r.get('best_pt_path'))}|{r.get('train_mAP50','')}|{r.get('train_mAP50_95','')}|")
    OUT_MD.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('wrote', OUT_CSV)
    print('wrote', OUT_MD)
if __name__ == '__main__': main()
