
from pathlib import Path
import shutil
import math
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings('ignore')

INPUT_DIR = Path('/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter')
DOMAIN_CSV = INPUT_DIR / 'domain_filter_results.csv'
OUT_DIR = Path('/root/autodl-tmp/road_damage_exp/screening/final_ours_200')
IMAGES_OUT = OUT_DIR / 'images'
LABELS_OUT = OUT_DIR / 'labels'
VIS_DIR = OUT_DIR / 'vis'
CLASSES = ['D00', 'D10', 'D20', 'D40']
CLASS_ID = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}
TARGET_PER_CLASS = 50
IMG_SIZE = 224
SELECTION_METHOD = 'Greedy Farthest Point Sampling with Quality Score'


def minmax(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr)); hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.ones_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def load_rgb_tensor(path, size=IMG_SIZE):
    img = Image.open(path).convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1)
    ten = ten * 2.0 - 1.0
    return ten


def setup_feature_model(device):
    try:
        import lpips
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lpips'])
        import lpips
    try:
        model = lpips.LPIPS(net='alex').to(device).eval()
        return 'lpips_alex', model, None
    except Exception as e:
        print(f'WARNING LPIPS init failed, falling back to torchvision resnet50 features: {e}', flush=True)
        try:
            import torchvision
            from torchvision.models import resnet50, ResNet50_Weights
            try:
                weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=weights)
                backbone = 'torchvision_resnet50_imagenet_features'
            except Exception as ee:
                print(f'WARNING pretrained ResNet50 unavailable, using weights=None: {ee}', flush=True)
                model = resnet50(weights=None)
                backbone = 'torchvision_resnet50_untrained_features'
            model.fc = torch.nn.Identity()
            model = model.to(device).eval()
            return backbone, None, model
        except Exception as ee:
            raise RuntimeError(f'Both LPIPS and torchvision fallback failed: {ee}')


def compute_distance_matrix(paths, backbone, lpips_model, resnet_model, device):
    n = len(paths)
    tensors = torch.stack([load_rgb_tensor(p) for p in tqdm(paths, desc='load_tensors')], dim=0)
    dist = np.zeros((n, n), dtype=np.float32)
    if backbone.startswith('lpips'):
        with torch.no_grad():
            for i in tqdm(range(n), desc='lpips_matrix'):
                ref = tensors[i:i+1].to(device)
                batch_size = 32 if device.type == 'cuda' else 8
                vals = []
                for j in range(0, n, batch_size):
                    cand = tensors[j:j+batch_size].to(device)
                    ref_rep = ref.expand(cand.shape[0], -1, -1, -1)
                    d = lpips_model(ref_rep, cand).view(-1).detach().cpu().numpy()
                    vals.append(d)
                dist[i, :] = np.concatenate(vals).astype(np.float32)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        return dist

    # ResNet fallback: convert [-1,1] to ImageNet normalized [0,1].
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for j in tqdm(range(0, n, 32), desc='resnet_features'):
            x = (tensors[j:j+32] + 1.0) / 2.0
            x = (x - mean) / std
            f = resnet_model(x.to(device)).detach().cpu()
            f = F.normalize(f, dim=1)
            feats.append(f)
    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)
    sim = feats @ feats.T
    dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim)).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def select_class(cdf, dist_matrix):
    n = len(cdf)
    if n < TARGET_PER_CLASS:
        raise RuntimeError(f'{cdf.iloc[0]["class_name"]} has only {n} candidates, need {TARGET_PER_CLASS}')
    domain = cdf['domain_score'].astype(float).to_numpy()
    norm_domain_all = minmax(domain)
    selected = [int(np.argmax(domain))]
    unselected = set(range(n)) - set(selected)
    records = []
    records.append({
        'idx': selected[0],
        'diversity_score': 0.0,
        'normalized_domain_score': float(norm_domain_all[selected[0]]),
        'normalized_diversity_score': 0.0,
        'final_score': float(norm_domain_all[selected[0]]),
        'selected_rank': 1,
    })
    while len(selected) < TARGET_PER_CLASS:
        cand = sorted(unselected)
        div_scores = np.array([float(np.min(dist_matrix[i, selected])) for i in cand], dtype=np.float32)
        norm_div = minmax(div_scores)
        norm_domain = norm_domain_all[cand]
        final = 0.5 * norm_domain + 0.5 * norm_div
        best_pos = int(np.argmax(final))
        best_i = cand[best_pos]
        selected.append(best_i)
        unselected.remove(best_i)
        records.append({
            'idx': best_i,
            'diversity_score': float(div_scores[best_pos]),
            'normalized_domain_score': float(norm_domain[best_pos]),
            'normalized_diversity_score': float(norm_div[best_pos]),
            'final_score': float(final[best_pos]),
            'selected_rank': len(selected),
        })
    return records


def output_name(class_name, src_name):
    p = Path(src_name)
    clean = src_name
    if not clean.startswith(class_name + '_'):
        clean = class_name + '_' + clean
    return clean


def copy_selected(row, out_img_name):
    src_img = Path(row['image_path'])
    src_label = Path(row['label_path'])
    out_img = IMAGES_OUT / out_img_name
    out_label = LABELS_OUT / (Path(out_img_name).stem + '.txt')
    shutil.copy2(src_img, out_img)
    shutil.copy2(src_label, out_label)
    return out_img, out_label


def validate_outputs(meta):
    img_files = sorted([p for p in IMAGES_OUT.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    label_files = sorted([p for p in LABELS_OUT.iterdir() if p.is_file() and p.suffix.lower() == '.txt'])
    img_stems = {p.stem for p in img_files}
    label_stems = {p.stem for p in label_files}
    errors = []
    if len(img_files) != 200:
        errors.append(f'image count {len(img_files)} != 200')
    if len(label_files) != 200:
        errors.append(f'label count {len(label_files)} != 200')
    if img_stems != label_stems:
        errors.append('image/label stems mismatch')
    for label in label_files:
        text = label.read_text(encoding='utf-8', errors='ignore').strip()
        if not text:
            errors.append(f'empty label: {label.name}')
            continue
        prefix = label.name.split('_', 1)[0]
        expected = CLASS_ID.get(prefix)
        if expected is None:
            errors.append(f'unknown class prefix: {label.name}')
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                errors.append(f'invalid class id {label.name}:{ln}')
                continue
            if cid != expected:
                errors.append(f'class id mismatch {label.name}:{ln} got {cid}, expected {expected}')
    return errors, len(img_files), len(label_files), img_stems == label_stems


def make_contact_sheet(cdf, out_path):
    rows = cdf.sort_values('selected_rank').to_dict('records')
    cell_w, cell_h = 180, 168
    cols = 10
    rows_n = math.ceil(len(rows) / cols)
    sheet = Image.new('RGB', (cols * cell_w, rows_n * cell_h + 28), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f'{cdf.iloc[0]["class_name"]} selected samples (n={len(rows)})', fill=(0, 0, 0))
    for i, r in enumerate(rows):
        img = Image.open(r['output_image_path']).convert('RGB')
        img.thumbnail((cell_w, cell_h - 34))
        x0 = (i % cols) * cell_w
        y0 = 28 + (i // cols) * cell_h
        sheet.paste(img, (x0 + (cell_w - img.width)//2, y0))
        draw.text((x0 + 4, y0 + cell_h - 30), f"#{int(r['selected_rank'])} d={float(r['domain_score']):.3f}\nv={float(r['diversity_score']):.3f}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in [IMAGES_OUT, LABELS_OUT, VIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for p in d.rglob('*') if d == VIS_DIR else d.glob('*'):
            if p.is_file():
                p.unlink()
    for p in list(IMAGES_OUT.glob('*')) + list(LABELS_OUT.glob('*')):
        if p.is_file():
            p.unlink()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    backbone, lpips_model, resnet_model = setup_feature_model(device)
    print(f'feature_backbone={backbone}', flush=True)

    df = pd.read_csv(DOMAIN_CSV)
    df = df[df['domain_pass'].astype(str).str.lower().isin(['true', '1', 'yes'])].copy()
    all_meta = []

    for class_name in CLASSES:
        cdf = df[df['class_name'] == class_name].copy().sort_values('domain_score', ascending=False).reset_index(drop=True)
        print(f'{class_name}: candidates={len(cdf)}', flush=True)
        if len(cdf) < TARGET_PER_CLASS:
            raise RuntimeError(f'{class_name} has {len(cdf)} candidates, need {TARGET_PER_CLASS}')
        paths = [Path(p) for p in cdf['image_path'].tolist()]
        dist = compute_distance_matrix(paths, backbone, lpips_model, resnet_model, device)
        selected_records = select_class(cdf, dist)
        class_rows = []
        for rec in selected_records:
            src = cdf.iloc[rec['idx']].copy()
            out_img_name = output_name(class_name, src['image_name'])
            out_img, out_label = copy_selected(src, out_img_name)
            row = {
                'class_name': class_name,
                'source_image_name': src['image_name'],
                'output_image_name': out_img.name,
                'source_image_path': src['image_path'],
                'source_label_path': src['label_path'],
                'output_image_path': str(out_img),
                'output_label_path': str(out_label),
                'domain_score': float(src['domain_score']),
                'normalized_domain_score': rec['normalized_domain_score'],
                'diversity_score': rec['diversity_score'],
                'normalized_diversity_score': rec['normalized_diversity_score'],
                'final_score': rec['final_score'],
                'selected_rank': rec['selected_rank'],
                'feature_backbone': backbone,
                'selection_method': SELECTION_METHOD,
            }
            class_rows.append(row)
            all_meta.append(row)
        class_df = pd.DataFrame(class_rows).sort_values('selected_rank')
        class_df.to_csv(OUT_DIR / f'{class_name}_selected.csv', index=False)
        make_contact_sheet(class_df, VIS_DIR / f'{class_name}_selected_contact_sheet.jpg')
        print(f'{class_name}: selected={len(class_df)}', flush=True)

    meta = pd.DataFrame(all_meta)
    meta_path = OUT_DIR / 'metadata_ours_200.csv'
    meta.to_csv(meta_path, index=False)
    errors, img_count, label_count, pair_match = validate_outputs(meta)
    success = not errors
    (OUT_DIR / 'ready_for_yolo_dataset.txt').write_text(f'ready_for_yolo_dataset = {success}\n', encoding='utf-8')
    if errors:
        err_text = '\n'.join(errors[:100])
        (OUT_DIR / 'ours_200_validation_errors.txt').write_text(err_text + '\n', encoding='utf-8')
        raise RuntimeError('Validation failed:\n' + err_text)

    summary_lines = [
        '===== Ours-200 LPIPS Diversity Selection Summary =====',
        f'Input candidate dir: {INPUT_DIR}',
        f'Domain result CSV: {DOMAIN_CSV}',
        f'Output dir: {OUT_DIR}',
        f'Feature backbone: {backbone}',
        f'Device: {device}',
        f'Selection method: {SELECTION_METHOD}',
        '',
    ]
    for class_name in CLASSES:
        cmeta = meta[meta['class_name'] == class_name]
        cinput = int(df[df['class_name'] == class_name].shape[0])
        summary_lines.extend([
            f'{class_name}: input_candidates={cinput}, selected={len(cmeta)}',
            f'{class_name}: domain_score mean={cmeta["domain_score"].mean():.6f}, min={cmeta["domain_score"].min():.6f}, max={cmeta["domain_score"].max():.6f}',
            f'{class_name}: diversity_score mean={cmeta["diversity_score"].mean():.6f}, min={cmeta["diversity_score"].min():.6f}, max={cmeta["diversity_score"].max():.6f}',
        ])
    summary_lines.extend([
        '',
        f'Total images: {img_count}',
        f'Total labels: {label_count}',
        f'Images/labels one-to-one: {pair_match}',
        'Label class check: passed',
        f'Successfully generated Ours-200: {success}',
    ])
    summary = '\n'.join(summary_lines) + '\n'
    (OUT_DIR / 'ours_200_summary.txt').write_text(summary, encoding='utf-8')
    print(summary, flush=True)
    print('metadata_preview:')
    print(meta.head().to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
