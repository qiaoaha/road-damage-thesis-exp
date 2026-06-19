
from pathlib import Path
import shutil
import random
import math
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings('ignore')

INPUT_ROOT = Path('/root/autodl-tmp/road_damage_exp/screening/four_class_structure_candidates')
RANDOM_OUT = Path('/root/autodl-tmp/road_damage_exp/screening/random_200')
LPIPS_OUT = Path('/root/autodl-tmp/road_damage_exp/screening/lpips_200')
CLASSES = ['D00', 'D10', 'D20', 'D40']
CLASS_ID = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}
TARGET_PER_CLASS = 50
SEED = 42
IMG_EXTS = {'.jpg', '.jpeg', '.png'}
IMG_SIZE = 224


def list_pairs(class_name):
    img_dir = INPUT_ROOT / class_name / 'images'
    lbl_dir = INPUT_ROOT / class_name / 'labels'
    rows = []
    for img in sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]):
        lbl = lbl_dir / (img.stem + '.txt')
        if lbl.exists():
            rows.append({'class_name': class_name, 'image_path': img, 'label_path': lbl, 'image_name': img.name})
    return rows


def reset_out(out):
    for sub in ['images', 'labels']:
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob('*'):
            if p.is_file():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)


def output_name(class_name, src_name):
    return src_name if src_name.startswith(class_name + '_') else f'{class_name}_{src_name}'


def copy_pair(row, out, out_img_name):
    out_img = out / 'images' / out_img_name
    out_lbl = out / 'labels' / (Path(out_img_name).stem + '.txt')
    shutil.copy2(row['image_path'], out_img)
    shutil.copy2(row['label_path'], out_lbl)
    return out_img, out_lbl


def validate_dataset(out):
    imgs = sorted([p for p in (out/'images').iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    lbls = sorted([p for p in (out/'labels').iterdir() if p.is_file() and p.suffix.lower()=='.txt'])
    errors=[]
    if len(imgs) != 200: errors.append(f'image count {len(imgs)} != 200')
    if len(lbls) != 200: errors.append(f'label count {len(lbls)} != 200')
    if {p.stem for p in imgs} != {p.stem for p in lbls}: errors.append('image/label stems mismatch')
    for lbl in lbls:
        text = lbl.read_text(encoding='utf-8', errors='ignore').strip()
        if not text:
            errors.append(f'empty label: {lbl.name}'); continue
        prefix = lbl.name.split('_',1)[0]
        expected = CLASS_ID.get(prefix)
        if expected is None:
            errors.append(f'unknown label prefix: {lbl.name}'); continue
        for ln, line in enumerate(text.splitlines(), 1):
            parts=line.split()
            if len(parts) < 5:
                errors.append(f'bad label line: {lbl.name}:{ln}'); continue
            try: cid=int(float(parts[0]))
            except Exception:
                errors.append(f'invalid class id: {lbl.name}:{ln}'); continue
            if cid != expected:
                errors.append(f'class id mismatch: {lbl.name}:{ln} got {cid}, expected {expected}')
    return errors, len(imgs), len(lbls), ({p.stem for p in imgs} == {p.stem for p in lbls})


def write_ready(out, ok):
    (out/'ready_for_yolo_dataset.txt').write_text(f'ready_for_yolo_dataset = {ok}\n', encoding='utf-8')


def build_random():
    reset_out(RANDOM_OUT)
    rng = random.Random(SEED)
    meta=[]; input_counts={}; selected_counts={}
    for cls in CLASSES:
        pairs = list_pairs(cls)
        input_counts[cls]=len(pairs)
        if len(pairs) < TARGET_PER_CLASS:
            raise RuntimeError(f'{cls} has only {len(pairs)} candidates')
        selected = rng.sample(pairs, TARGET_PER_CLASS)
        selected_counts[cls]=len(selected)
        for rank, row in enumerate(selected, 1):
            out_name = output_name(cls, row['image_name'])
            out_img, out_lbl = copy_pair(row, RANDOM_OUT, out_name)
            meta.append({
                'class_name': cls,
                'source_image_name': row['image_name'],
                'output_image_name': out_img.name,
                'source_image_path': str(row['image_path']),
                'source_label_path': str(row['label_path']),
                'output_image_path': str(out_img),
                'output_label_path': str(out_lbl),
                'selected_rank': rank,
                'selection_method': 'class_balanced_random_sampling',
                'seed': SEED,
            })
    meta_df=pd.DataFrame(meta)
    meta_df.to_csv(RANDOM_OUT/'metadata_random_200.csv', index=False)
    errors,img_count,lbl_count,pair_match=validate_dataset(RANDOM_OUT)
    ok=not errors
    write_ready(RANDOM_OUT, ok)
    if errors:
        (RANDOM_OUT/'validation_errors.txt').write_text('\n'.join(errors)+'\n', encoding='utf-8')
        raise RuntimeError('Random-200 validation failed:\n'+'\n'.join(errors[:20]))
    lines=['===== Random-200 Summary =====', f'Input root: {INPUT_ROOT}', f'Output dir: {RANDOM_OUT}', 'selection_method: class_balanced_random_sampling', f'seed: {SEED}', '']
    for cls in CLASSES:
        lines.append(f'{cls}: input_candidates={input_counts[cls]}, selected={selected_counts[cls]}')
    lines += ['', f'Total images: {img_count}', f'Total labels: {lbl_count}', f'pair_match: {pair_match}', 'label_class_check: passed', 'ready_for_yolo_dataset = True']
    (RANDOM_OUT/'random_200_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return meta_df, input_counts


def load_rgb_tensor(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2,0,1)
    return ten * 2.0 - 1.0


def setup_feature(device):
    try:
        import lpips
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lpips'])
        import lpips
    try:
        model=lpips.LPIPS(net='alex').to(device).eval()
        return 'lpips_alex', model, None
    except Exception as e:
        print(f'WARNING LPIPS init failed, fallback to resnet50: {e}', flush=True)
        from torchvision.models import resnet50, ResNet50_Weights
        try:
            model=resnet50(weights=ResNet50_Weights.DEFAULT); backbone='torchvision_resnet50_imagenet_features'
        except Exception as ee:
            print(f'WARNING pretrained resnet unavailable: {ee}', flush=True)
            model=resnet50(weights=None); backbone='torchvision_resnet50_untrained_features'
        model.fc=torch.nn.Identity(); model=model.to(device).eval()
        return backbone, None, model


def distance_matrix(paths, backbone, lpips_model, resnet_model, device):
    n=len(paths)
    tensors=torch.stack([load_rgb_tensor(p) for p in tqdm(paths, desc='load_tensors')])
    if backbone.startswith('lpips'):
        dist=np.zeros((n,n), dtype=np.float32)
        with torch.no_grad():
            for i in tqdm(range(n), desc='lpips_matrix'):
                ref=tensors[i:i+1].to(device)
                vals=[]; bs=32 if device.type=='cuda' else 8
                for j in range(0,n,bs):
                    cand=tensors[j:j+bs].to(device)
                    d=lpips_model(ref.expand(cand.shape[0],-1,-1,-1), cand).view(-1).detach().cpu().numpy()
                    vals.append(d)
                dist[i,:]=np.concatenate(vals).astype(np.float32)
                if device.type=='cuda': torch.cuda.empty_cache()
        dist=(dist+dist.T)/2; np.fill_diagonal(dist,0); return dist
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    feats=[]
    with torch.no_grad():
        for j in tqdm(range(0,n,32), desc='resnet_features'):
            x=(tensors[j:j+32]+1)/2; x=(x-mean)/std
            f=F.normalize(resnet_model(x.to(device)).detach().cpu(), dim=1)
            feats.append(f)
    feats=torch.cat(feats).numpy().astype(np.float32)
    sim=feats@feats.T
    dist=np.sqrt(np.maximum(0,2-2*sim)).astype(np.float32); np.fill_diagonal(dist,0); return dist


def farthest_select(dist):
    n=dist.shape[0]
    selected=[0]
    unselected=set(range(1,n))
    records=[{'idx':0,'diversity_score':0.0,'selected_rank':1}]
    while len(selected)<TARGET_PER_CLASS:
        cand=sorted(unselected)
        div=np.array([float(np.min(dist[i, selected])) for i in cand], dtype=np.float32)
        pos=int(np.argmax(div)); idx=cand[pos]
        selected.append(idx); unselected.remove(idx)
        records.append({'idx':idx,'diversity_score':float(div[pos]),'selected_rank':len(selected)})
    return records


def build_lpips():
    reset_out(LPIPS_OUT)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, lpips_model, resnet_model = setup_feature(device)
    meta=[]; input_counts={}; selected_counts={}
    for cls in CLASSES:
        pairs=list_pairs(cls)
        input_counts[cls]=len(pairs)
        if len(pairs)<TARGET_PER_CLASS: raise RuntimeError(f'{cls} has only {len(pairs)} candidates')
        print(f'{cls}: candidates={len(pairs)}', flush=True)
        paths=[r['image_path'] for r in pairs]
        dist=distance_matrix(paths, backbone, lpips_model, resnet_model, device)
        selected=farthest_select(dist)
        selected_counts[cls]=len(selected)
        for rec in selected:
            row=pairs[rec['idx']]
            out_name=output_name(cls, row['image_name'])
            out_img,out_lbl=copy_pair(row, LPIPS_OUT, out_name)
            meta.append({
                'class_name': cls,
                'source_image_name': row['image_name'],
                'output_image_name': out_img.name,
                'source_image_path': str(row['image_path']),
                'source_label_path': str(row['label_path']),
                'output_image_path': str(out_img),
                'output_label_path': str(out_lbl),
                'diversity_score': rec['diversity_score'],
                'selected_rank': rec['selected_rank'],
                'feature_backbone': backbone,
                'selection_method': 'Greedy Farthest Point Sampling',
            })
    meta_df=pd.DataFrame(meta)
    meta_df.to_csv(LPIPS_OUT/'metadata_lpips_200.csv', index=False)
    errors,img_count,lbl_count,pair_match=validate_dataset(LPIPS_OUT)
    ok=not errors
    write_ready(LPIPS_OUT, ok)
    if errors:
        (LPIPS_OUT/'validation_errors.txt').write_text('\n'.join(errors)+'\n', encoding='utf-8')
        raise RuntimeError('LPIPS-200 validation failed:\n'+'\n'.join(errors[:20]))
    lines=['===== LPIPS-200 Summary =====', f'Input root: {INPUT_ROOT}', f'Output dir: {LPIPS_OUT}', f'feature_backbone: {backbone}', f'device: {device}', 'selection_method: Greedy Farthest Point Sampling', '']
    for cls in CLASSES:
        c=meta_df[meta_df.class_name==cls]
        lines.append(f'{cls}: input_candidates={input_counts[cls]}, selected={selected_counts[cls]}, diversity_mean={c.diversity_score.mean():.6f}, diversity_min={c.diversity_score.min():.6f}, diversity_max={c.diversity_score.max():.6f}')
    lines += ['', f'Total images: {img_count}', f'Total labels: {lbl_count}', f'pair_match: {pair_match}', 'label_class_check: passed', 'ready_for_yolo_dataset = True']
    (LPIPS_OUT/'lpips_200_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return meta_df, input_counts, backbone


def main():
    print('Building Random-200...', flush=True)
    random_meta,_ = build_random()
    print('Random-200 done', flush=True)
    print('Building LPIPS-200...', flush=True)
    lpips_meta,_,backbone = build_lpips()
    print('LPIPS-200 done', flush=True)
    print((RANDOM_OUT/'random_200_summary.txt').read_text(encoding='utf-8'))
    print((LPIPS_OUT/'lpips_200_summary.txt').read_text(encoding='utf-8'))

if __name__ == '__main__':
    main()
