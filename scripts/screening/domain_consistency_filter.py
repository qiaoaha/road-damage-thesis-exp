
from pathlib import Path
import shutil
import math
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

warnings.filterwarnings('ignore')

INPUT_ROOT = Path('/root/autodl-tmp/road_damage_exp/screening/four_class_structure_candidates')
REAL_IMG_DIR = Path('/root/autodl-tmp/road_damage_exp/processed/real_yolo/images/train')
REAL_LABEL_DIR = Path('/root/autodl-tmp/road_damage_exp/processed/real_yolo/labels/train')
OUT_ROOT = Path('/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter')
SCRIPT_PATH = Path('/root/autodl-tmp/road_damage_exp/domain_consistency_filter.py')

CLASSES = ['D00', 'D10', 'D20', 'D40']
KEEP_COUNTS = {'D00': 134, 'D10': 65, 'D20': 91, 'D40': 58}
IMG_EXTS = {'.png', '.jpg', '.jpeg'}
MAX_SIDE_FEATURE = 768
LBP_P = 8
LBP_R = 1
LBP_BINS = LBP_P + 2
GLCM_LEVELS = 32
VIS_PER_SHEET = 16


def list_images(path):
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]) if path.exists() else []


def read_label_boxes(label_path, width, height):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            _, xc, yc, bw, bh = parts[:5]
            xc, yc, bw, bh = map(float, [xc, yc, bw, bh])
        except Exception:
            continue
        x1 = int(round((xc - bw / 2) * width))
        y1 = int(round((yc - bh / 2) * height))
        x2 = int(round((xc + bw / 2) * width))
        y2 = int(round((yc + bh / 2) * height))
        x1 = max(0, min(width, x1)); x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1)); y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def resize_for_feature(img_bgr, boxes):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, MAX_SIDE_FEATURE / max(h, w))
    if scale >= 1.0:
        return img_bgr, boxes
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        new_boxes.append((int(round(x1 * scale)), int(round(y1 * scale)), int(round(x2 * scale)), int(round(y2 * scale))))
    return resized, new_boxes


def background_mask(shape, boxes):
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        pad_x = max(2, int((x2 - x1) * 0.08))
        pad_y = max(2, int((y2 - y1) * 0.08))
        xx1 = max(0, x1 - pad_x); yy1 = max(0, y1 - pad_y)
        xx2 = min(w, x2 + pad_x); yy2 = min(h, y2 + pad_y)
        mask[yy1:yy2, xx1:xx2] = False
    if mask.mean() < 0.15:
        mask[:] = True
    return mask


def masked_values(channel, mask):
    vals = channel[mask]
    if vals.size == 0:
        vals = channel.reshape(-1)
    return vals.astype(np.float32)


def feature_for_image(image_path, label_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Cannot read image: {image_path}')
    h0, w0 = img.shape[:2]
    boxes = read_label_boxes(label_path, w0, h0)
    img, boxes = resize_for_feature(img, boxes)
    mask = background_mask(img.shape, boxes)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lab_feats = []
    for c in range(3):
        vals = masked_values(lab[:, :, c], mask)
        lab_feats.extend([float(vals.mean()), float(vals.std())])

    v_vals = masked_values(hsv[:, :, 2], mask)
    light_feats = [float(v_vals.mean()), float(v_vals.std())]

    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method='uniform')
    lbp_vals = lbp[mask]
    if lbp_vals.size == 0:
        lbp_vals = lbp.reshape(-1)
    hist, _ = np.histogram(lbp_vals, bins=np.arange(LBP_BINS + 1), range=(0, LBP_BINS), density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)

    # Compute GLCM on a small background-filled grayscale canvas for speed and stability.
    gray_small = gray
    mask_small = mask
    if max(gray.shape) > 384:
        scale = 384 / max(gray.shape)
        gray_small = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask.astype(np.uint8), (gray_small.shape[1], gray_small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    bg_mean = int(np.mean(gray_small[mask_small])) if mask_small.any() else int(np.mean(gray_small))
    glcm_img = gray_small.copy()
    glcm_img[~mask_small] = bg_mean
    quant = np.clip((glcm_img.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8), 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(quant, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=GLCM_LEVELS, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    energy = float(graycoprops(glcm, 'energy').mean())

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vals = lap[mask]
    if lap_vals.size == 0:
        lap_vals = lap.reshape(-1)
    sharpness = float(np.var(lap_vals))

    return {
        'color': np.array(lab_feats, dtype=np.float32),
        'light': np.array(light_feats, dtype=np.float32),
        'texture': np.concatenate([hist, np.array([contrast, homogeneity, energy], dtype=np.float32)]),
        'sharpness': np.array([sharpness], dtype=np.float32),
    }


def safe_stats(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-6


def zdist(vec, mean, std):
    return float(np.sqrt(np.mean(((vec - mean) / std) ** 2)))


def minmax_norm(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def copy_pair(src_img, src_label, dst_img_dir, dst_label_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img_dir / src_img.name)
    shutil.copy2(src_label, dst_label_dir / src_label.name)


def make_contact_sheet(rows, out_path, title):
    rows = list(rows)[:VIS_PER_SHEET]
    if not rows:
        return
    thumbs = []
    cell_w, cell_h = 220, 190
    header_h = 34
    for r in rows:
        img = Image.open(r['image_path']).convert('RGB')
        img.thumbnail((cell_w, cell_h - 42))
        canvas = Image.new('RGB', (cell_w, cell_h), 'white')
        x = (cell_w - img.width) // 2
        canvas.paste(img, (x, 0))
        draw = ImageDraw.Draw(canvas)
        txt = f"{Path(r['image_path']).name[:24]}\nscore={r['domain_score']:.3f} rank={int(r['domain_rank_in_class'])}"
        draw.text((6, cell_h - 38), txt, fill=(0, 0, 0))
        thumbs.append(canvas)
    cols = 4
    rows_n = math.ceil(len(thumbs) / cols)
    sheet = Image.new('RGB', (cols * cell_w, rows_n * cell_h + header_h), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = header_h + (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    vis_root = OUT_ROOT / 'vis'
    for class_key in CLASSES:
        for sub in ['images', 'labels']:
            d = OUT_ROOT / class_key / sub
            d.mkdir(parents=True, exist_ok=True)
            for p in d.glob('*'):
                if p.is_file():
                    p.unlink()
    if vis_root.exists():
        for p in vis_root.rglob('*'):
            if p.is_file():
                p.unlink()
    vis_root.mkdir(parents=True, exist_ok=True)

    print('Collecting real road background features...', flush=True)
    real_images = list_images(REAL_IMG_DIR)
    print(f'real_train_images={len(real_images)}', flush=True)
    real_features = {'color': [], 'light': [], 'texture': [], 'sharpness': []}
    bad_real = 0
    for img_path in tqdm(real_images, desc='real_features'):
        label_path = REAL_LABEL_DIR / (img_path.stem + '.txt')
        try:
            feat = feature_for_image(img_path, label_path)
            for k in real_features:
                real_features[k].append(feat[k])
        except Exception as e:
            bad_real += 1
            print(f'WARNING real feature failed: {img_path} | {e}', flush=True)
    proto = {}
    for k, vals in real_features.items():
        if not vals:
            raise RuntimeError(f'No real features for {k}')
        proto[k] = safe_stats(np.vstack(vals))
    print(f'real_bad_count={bad_real}', flush=True)

    print('Scoring generated structure candidates...', flush=True)
    rows = []
    input_counts = {}
    input_pair = {}
    for class_key in CLASSES:
        img_dir = INPUT_ROOT / class_key / 'images'
        lbl_dir = INPUT_ROOT / class_key / 'labels'
        imgs = list_images(img_dir)
        lbl_stems = {p.stem for p in lbl_dir.glob('*.txt')}
        img_stems = {p.stem for p in imgs}
        input_counts[class_key] = len(imgs)
        input_pair[class_key] = img_stems == lbl_stems
        for img_path in tqdm(imgs, desc=f'{class_key}_features'):
            label_path = lbl_dir / (img_path.stem + '.txt')
            feat = feature_for_image(img_path, label_path)
            color_dist = zdist(feat['color'], *proto['color'])
            texture_dist = zdist(feat['texture'], *proto['texture'])
            light_dist = zdist(feat['light'], *proto['light'])
            sharpness_dist = zdist(feat['sharpness'], *proto['sharpness'])
            rows.append({
                'image_name': img_path.name,
                'class_name': class_key,
                'image_path': str(img_path),
                'label_path': str(label_path),
                'color_dist': color_dist,
                'texture_dist': texture_dist,
                'light_dist': light_dist,
                'sharpness_dist': sharpness_dist,
            })
    df = pd.DataFrame(rows)
    for col in ['color_dist', 'texture_dist', 'light_dist', 'sharpness_dist']:
        df[col + '_norm'] = minmax_norm(df[col].values)
    dist_norm_cols = ['color_dist_norm', 'texture_dist_norm', 'light_dist_norm', 'sharpness_dist_norm']
    df['domain_score'] = 1.0 - df[dist_norm_cols].mean(axis=1)
    df['domain_score'] = df['domain_score'].clip(0.0, 1.0)
    df['domain_rank_in_class'] = df.groupby('class_name')['domain_score'].rank(method='first', ascending=False).astype(int)
    df['domain_pass'] = False
    for class_key in CLASSES:
        keep_n = KEEP_COUNTS[class_key]
        class_idx = df[df['class_name'] == class_key].sort_values('domain_score', ascending=False).head(keep_n).index
        df.loc[class_idx, 'domain_pass'] = True

    # Copy retained pairs.
    for _, r in df[df['domain_pass']].iterrows():
        class_key = r['class_name']
        copy_pair(Path(r['image_path']), Path(r['label_path']), OUT_ROOT / class_key / 'images', OUT_ROOT / class_key / 'labels')

    result_cols = ['image_name', 'class_name', 'image_path', 'label_path', 'color_dist', 'texture_dist', 'light_dist', 'sharpness_dist', 'domain_score', 'domain_rank_in_class', 'domain_pass']
    result_csv = OUT_ROOT / 'domain_filter_results.csv'
    df[result_cols].sort_values(['class_name', 'domain_rank_in_class']).to_csv(result_csv, index=False)

    summary_lines = []
    summary_rows = []
    all_ready = True
    for class_key in CLASSES:
        cdf = df[df['class_name'] == class_key]
        kept = cdf[cdf['domain_pass']]
        out_imgs = list_images(OUT_ROOT / class_key / 'images')
        out_lbls = sorted((OUT_ROOT / class_key / 'labels').glob('*.txt'))
        pair_match = {p.stem for p in out_imgs} == {p.stem for p in out_lbls}
        if len(out_imgs) < 50 or len(out_lbls) < 50 or not pair_match:
            all_ready = False
        summary_rows.append({
            'class': class_key,
            'input_count': int(input_counts[class_key]),
            'kept_count': int(len(kept)),
            'output_image_count': int(len(out_imgs)),
            'output_label_count': int(len(out_lbls)),
            'pair_match': bool(pair_match),
            'domain_score_mean': float(cdf['domain_score'].mean()) if len(cdf) else 0.0,
            'domain_score_min': float(cdf['domain_score'].min()) if len(cdf) else 0.0,
            'domain_score_max': float(cdf['domain_score'].max()) if len(cdf) else 0.0,
        })
        summary_lines.append(
            f"{class_key}: input={input_counts[class_key]}, kept={len(kept)}, images={len(out_imgs)}, labels={len(out_lbls)}, "
            f"pair_match={pair_match}, score_mean={summary_rows[-1]['domain_score_mean']:.6f}, "
            f"score_min={summary_rows[-1]['domain_score_min']:.6f}, score_max={summary_rows[-1]['domain_score_max']:.6f}"
        )

        class_vis = vis_root / class_key
        class_vis.mkdir(parents=True, exist_ok=True)
        ranked = cdf.sort_values('domain_score', ascending=False)
        make_contact_sheet(ranked.to_dict('records'), class_vis / 'top_domain_samples.jpg', f'{class_key} top domain samples')
        make_contact_sheet(ranked.tail(VIS_PER_SHEET).sort_values('domain_score').to_dict('records'), class_vis / 'low_domain_samples.jpg', f'{class_key} low domain samples')
        plt.figure(figsize=(6, 4))
        plt.hist(cdf['domain_score'], bins=20, color='#4C78A8', edgecolor='white')
        plt.title(f'{class_key} domain_score')
        plt.xlabel('domain_score')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(class_vis / 'domain_score_hist.png', dpi=160)
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_detail_csv = OUT_ROOT / 'domain_filter_summary_by_class.csv'
    summary_df.to_csv(summary_detail_csv, index=False)
    summary_text = '\n'.join([
        '===== Domain Consistency Filter Summary =====',
        f'Input root: {INPUT_ROOT}',
        f'Real image dir: {REAL_IMG_DIR}',
        f'Real label dir: {REAL_LABEL_DIR}',
        f'Real train images used: {len(real_images)}',
        f'Real feature failures: {bad_real}',
        f'Results CSV: {result_csv}',
        f'By-class summary CSV: {summary_detail_csv}',
        '',
        *summary_lines,
        '',
        f'all_classes_retained_ge_50 = {all_ready}',
        f'ready_for_diversity_filter = {all_ready}',
    ]) + '\n'
    (OUT_ROOT / 'domain_filter_summary.txt').write_text(summary_text, encoding='utf-8')
    (OUT_ROOT / 'ready_for_diversity_filter.txt').write_text(f'ready_for_diversity_filter = {all_ready}\n', encoding='utf-8')
    print(summary_text, flush=True)
    print('Preview:')
    print(df[result_cols].sort_values(['class_name', 'domain_rank_in_class']).head().to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
