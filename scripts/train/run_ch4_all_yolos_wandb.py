#!/usr/bin/env python3
import os, sys, csv, json, time, subprocess, shutil
from pathlib import Path

BASE = Path('/root/autodl-tmp/road_damage_exp')
DATA_ROOT = BASE / 'datasets_yolo_ch4_base80'
RUN_ROOT = BASE / 'runs_ch4_base80_wandb'
LOG_DIR = RUN_ROOT / 'logs'
STATUS_PATH = RUN_ROOT / 'ch4_training_status.jsonl'
PROJECT = 'road_damage_ch4_base80_yolo_compare'
CACHE = Path('/root/autodl-tmp/cache')
YOLOV5_REPO = Path('/root/yolov5')
PYTHON = sys.executable

CLASSES = ['D00_longitudinal_crack','D10_transverse_crack','D20_alligator_crack','D40_pothole']
DATASETS = {
    'base80_only': DATA_ROOT / 'base80_only',
    'base80_plus_random_200': DATA_ROOT / 'base80_plus_random_200',
    'base80_plus_lpips_200': DATA_ROOT / 'base80_plus_lpips_200',
    'base80_plus_ours_200': DATA_ROOT / 'base80_plus_ours_200',
}
FAMILIES = {
    'yolov5': {'weights': BASE / 'weights/yolov5s.pt'},
    'yolov8': {'weights': BASE / 'yolov8s.pt'},
    'yolov11': {'weights': BASE / 'yolo11s.pt'},
}

def env_for(run_name):
    env = os.environ.copy()
    env.update({
        'HF_HOME': str(CACHE / 'huggingface'),
        'TORCH_HOME': str(CACHE / 'torch'),
        'XDG_CACHE_HOME': str(CACHE),
        'WANDB_PROJECT': PROJECT,
        'WANDB_NAME': run_name,
        'WANDB_MODE': env.get('WANDB_MODE', 'online'),
        'WANDB_SILENT': 'true',
        'PYTHONUNBUFFERED': '1',
    })
    return env

def run_cmd(cmd, log_path, cwd=None, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write('\n\n===== %s =====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
        f.write('CMD: ' + ' '.join(map(str, cmd)) + '\n')
        f.flush()
        p = subprocess.Popen(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None, env=env)
        code = p.wait()
    text = log_path.read_text(encoding='utf-8', errors='replace')[-20000:]
    return code, text

def has_wandb_login():
    try:
        p = subprocess.run(['wandb','status'], capture_output=True, text=True, timeout=30)
        txt = (p.stdout or '') + (p.stderr or '')
        return 'api_key' in txt and 'null' not in txt.split('api_key',1)[1].split('\n',1)[0]
    except Exception:
        return False

def validate_inputs():
    RUN_ROOT.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    if not YOLOV5_REPO.exists(): missing.append(str(YOLOV5_REPO))
    for fam, cfg in FAMILIES.items():
        if not cfg['weights'].exists(): missing.append(str(cfg['weights']))
    for name, path in DATASETS.items():
        if not (path/'ready_for_training.txt').exists(): missing.append(str(path/'ready_for_training.txt'))
        if not (path/'data.yaml').exists(): missing.append(str(path/'data.yaml'))
        if not (path/'data_yolov5.yaml').exists(): missing.append(str(path/'data_yolov5.yaml'))
    if missing:
        raise SystemExit('Missing required inputs:\n' + '\n'.join(missing))

def status(row):
    row = dict(row); row['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def experiment_done(fam, run_name):
    train_dir = RUN_ROOT / fam / run_name
    test_dir = RUN_ROOT / f'{fam}_test' / run_name
    best = train_dir / 'weights/best.pt'
    results = train_dir / 'results.csv'
    return best.exists() and (results.exists() or test_dir.exists())

def train_yolov5(dataset_name, dataset_path, batch):
    run_name = f'yolov5_{dataset_name}'
    if experiment_done('yolov5', run_name):
        status({'model_family':'yolov5','dataset_variant':dataset_name,'run_name':run_name,'status':'skipped_existing','batch':batch})
        return True, batch
    data = dataset_path / 'data_yolov5.yaml'
    env = env_for(run_name)
    train_cmd = [PYTHON, YOLOV5_REPO/'train.py', '--img', '640', '--batch', str(batch), '--epochs', '100',
                 '--data', data, '--weights', FAMILIES['yolov5']['weights'], '--device', '0', '--workers', '8',
                 '--project', RUN_ROOT/'yolov5', '--name', run_name, '--exist-ok', '--seed', '42', '--patience', '50']
    log = LOG_DIR / f'{run_name}.log'
    code, text = run_cmd(train_cmd, log, cwd=YOLOV5_REPO, env=env)
    if code != 0 and ('out of memory' in text.lower() or 'cuda oom' in text.lower()):
        return False, 8
    if code != 0:
        status({'model_family':'yolov5','dataset_variant':dataset_name,'run_name':run_name,'status':'failed_train','batch':batch,'log':str(log)})
        return True, batch
    best = RUN_ROOT/'yolov5'/run_name/'weights/best.pt'
    val_cmd = [PYTHON, YOLOV5_REPO/'val.py', '--img', '640', '--batch', str(batch), '--data', data, '--weights', best,
               '--device', '0', '--task', 'test', '--project', RUN_ROOT/'yolov5_test', '--name', run_name, '--exist-ok']
    code, _ = run_cmd(val_cmd, log, cwd=YOLOV5_REPO, env=env)
    status({'model_family':'yolov5','dataset_variant':dataset_name,'run_name':run_name,'status':'completed' if code==0 else 'failed_test','batch':batch,'best_pt':str(best),'log':str(log)})
    return True, batch

def train_ultra(fam, dataset_name, dataset_path, batch):
    run_name = f'{fam}_{dataset_name}'
    if experiment_done(fam, run_name):
        status({'model_family':fam,'dataset_variant':dataset_name,'run_name':run_name,'status':'skipped_existing','batch':batch})
        return True, batch
    env = env_for(run_name)
    data = dataset_path / 'data.yaml'
    log = LOG_DIR / f'{run_name}.log'
    train_cmd = ['yolo','detect','train', f'model={FAMILIES[fam]["weights"]}', f'data={data}', 'epochs=100', 'imgsz=640',
                 f'batch={batch}', 'seed=42', 'device=0', 'workers=8', 'pretrained=True', 'mosaic=1.0', 'close_mosaic=10',
                 'patience=50', 'plots=True', f'project={RUN_ROOT/fam}', f'name={run_name}', 'exist_ok=True']
    code, text = run_cmd(train_cmd, log, env=env)
    if code != 0 and ('out of memory' in text.lower() or 'cuda oom' in text.lower()):
        return False, 8
    if code != 0:
        status({'model_family':fam,'dataset_variant':dataset_name,'run_name':run_name,'status':'failed_train','batch':batch,'log':str(log)})
        return True, batch
    best = RUN_ROOT/fam/run_name/'weights/best.pt'
    val_cmd = ['yolo','detect','val', f'model={best}', f'data={data}', 'split=test', 'imgsz=640', f'batch={batch}',
               'device=0', 'plots=True', f'project={RUN_ROOT/(fam+"_test")}', f'name={run_name}', 'exist_ok=True']
    code, _ = run_cmd(val_cmd, log, env=env)
    status({'model_family':fam,'dataset_variant':dataset_name,'run_name':run_name,'status':'completed' if code==0 else 'failed_test','batch':batch,'best_pt':str(best),'log':str(log)})
    return True, batch

def main():
    validate_inputs()
    print('wandb_login_detected=', has_wandb_login())
    family_batch = {fam:16 for fam in FAMILIES}
    for fam in ['yolov5','yolov8','yolov11']:
        for dataset_name, dataset_path in DATASETS.items():
            while True:
                if fam == 'yolov5': ok, new_batch = train_yolov5(dataset_name, dataset_path, family_batch[fam])
                else: ok, new_batch = train_ultra(fam, dataset_name, dataset_path, family_batch[fam])
                if ok: break
                if new_batch < family_batch[fam]:
                    family_batch[fam] = new_batch
                    status({'model_family':fam,'dataset_variant':dataset_name,'status':'retry_after_oom','batch':new_batch})
                    continue
                break
    print('training loop finished')

if __name__ == '__main__':
    main()
