#!/usr/bin/env python3
import os, shutil, subprocess, time
from pathlib import Path
BASE=Path('/root/autodl-tmp/road_damage_exp')
REPO=BASE/'github_repo/road-damage-thesis-exp'
RUN_ROOT=BASE/'runs_ch4_base80_wandb'
DATA_ROOT=BASE/'datasets_yolo_ch4_base80'

def copy_file(src, dst):
    src=Path(src); dst=Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,dst); return True
    return False

def append(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f: f.write(text)

def main():
    os.chdir(REPO)
    copied=[]
    for s in ['build_ch4_base80_only_and_yolov5_yaml.py','run_ch4_all_yolos_wandb.py','collect_ch4_all_yolos_wandb_results.py','log_ch4_results_to_wandb.py','sync_ch4_training_to_github.py']:
        if copy_file(BASE/s, REPO/'scripts/train'/s): copied.append(s)
    for ds in ['base80_only','base80_plus_random_200','base80_plus_lpips_200','base80_plus_ours_200']:
        for f in ['data.yaml','data_yolov5.yaml','dataset_summary.txt','dataset_class_distribution.csv','ready_for_training.txt']:
            copy_file(DATA_ROOT/ds/f, REPO/'results/dataset_summary/ch4_base80'/ds/f)
            if f.endswith('.yaml'): copy_file(DATA_ROOT/ds/f, REPO/'configs/data_yaml/ch4_base80'/ds/f)
    for f in ['ch4_base80_dataset_summary.csv','ch4_base80_dataset_summary_with_base80_only.csv']:
        copy_file(DATA_ROOT/f, REPO/'results/dataset_summary/ch4_base80'/f)
    for f in ['ch4_training_status.jsonl','ch4_all_yolos_results.csv','ch4_all_yolos_results.md']:
        copy_file(RUN_ROOT/f, REPO/'results/train_summary/ch4_base80_wandb'/f)
    fw=REPO/'final_weights/README_weights.md'
    fw.parent.mkdir(parents=True, exist_ok=True)
    if not fw.exists(): fw.write_text('# Final weights registry\n\n', encoding='utf-8')
    append(fw, f"\n## Chapter 4 W&B run snapshot - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\nWeights are kept on the server unless explicitly selected as final thesis weights. Run root: `{RUN_ROOT}`.\n\n")
    append(REPO/'docs/path_registry.md', f"\n## Chapter 4 base80 YOLO datasets and W&B runs - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n- Correct dataset root: `{DATA_ROOT}`\n- Abandoned wrong full-real datasets: `/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_*`\n- W&B run root: `{RUN_ROOT}`\n- Results CSV: `{RUN_ROOT/'ch4_all_yolos_results.csv'}`\n")
    append(REPO/'docs/codex_operation_log.md', f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} Chapter 4 YOLO W&B training sync\n\n- Scripts: `{', '.join(copied)}`\n- Dataset root: `{DATA_ROOT}`\n- Run root: `{RUN_ROOT}`\n- GitHub sync: attempted by server-side script\n")
    subprocess.run(['git','status'], check=False)
    subprocess.run(['git','add','.'], check=True)
    diff=subprocess.run(['git','diff','--cached','--quiet'])
    if diff.returncode != 0:
        subprocess.run(['git','commit','-m','run chapter4 yolov5 yolov8 yolov11 wandb experiments'], check=True)
    else:
        print('nothing to commit')
    subprocess.run(['git','push'], check=True)
    print(subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
if __name__ == '__main__': main()
