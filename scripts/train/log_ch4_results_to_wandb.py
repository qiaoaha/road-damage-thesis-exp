#!/usr/bin/env python3
import csv, os
from pathlib import Path
BASE=Path('/root/autodl-tmp/road_damage_exp')
CSV=BASE/'runs_ch4_base80_wandb/ch4_all_yolos_results.csv'
def main():
    if not CSV.exists():
        print('missing', CSV); return 1
    try:
        import wandb
    except Exception as e:
        print('wandb import failed', e); return 1
    rows=list(csv.DictReader(CSV.open(encoding='utf-8')))
    run=wandb.init(project='road_damage_ch4_base80_yolo_compare', name='ch4_all_yolos_summary', job_type='summary')
    table=wandb.Table(columns=list(rows[0].keys()) if rows else ['empty'])
    for r in rows: table.add_data(*[r.get(c,'') for c in table.columns])
    wandb.log({'ch4_results': table, 'experiment_count': len(rows)})
    run.finish()
    print('logged summary to wandb')
if __name__ == '__main__': raise SystemExit(main() or 0)
