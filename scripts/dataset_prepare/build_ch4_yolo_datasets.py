
from pathlib import Path
import shutil
import pandas as pd
from collections import Counter

REAL_ROOT = Path('/root/autodl-tmp/road_damage_exp/processed/real_yolo')
AUG_SETS = {
    'real_plus_random_200': {'root': Path('/root/autodl-tmp/road_damage_exp/screening/random_200'), 'prefix': 'random_'},
    'real_plus_lpips_200': {'root': Path('/root/autodl-tmp/road_damage_exp/screening/lpips_200'), 'prefix': 'lpips_'},
    'real_plus_ours_200': {'root': Path('/root/autodl-tmp/road_damage_exp/screening/final_ours_200'), 'prefix': 'ours_'},
}
OUT_ROOT = Path('/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_NAMES = {0:'D00_longitudinal_crack',1:'D10_transverse_crack',2:'D20_alligator_crack',3:'D40_pothole'}


def detect_real_layout(root):
    fmt_a = all((root/split/sub).exists() for split in ['train','val','test'] for sub in ['images','labels'])
    fmt_b = all((root/sub/split).exists() for split in ['train','val','test'] for sub in ['images','labels'])
    if fmt_a:
        return 'A', {split:{'images':root/split/'images','labels':root/split/'labels'} for split in ['train','val','test']}
    if fmt_b:
        return 'B', {split:{'images':root/'images'/split,'labels':root/'labels'/split} for split in ['train','val','test']}
    raise RuntimeError(f'Cannot detect real_yolo layout under {root}')


def list_images(d):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def reset_dataset_dir(root):
    if root.exists():
        for p in sorted(root.rglob('*'), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
    for split in ['train','val','test']:
        (root/'images'/split).mkdir(parents=True, exist_ok=True)
        (root/'labels'/split).mkdir(parents=True, exist_ok=True)


def copy_real_split(src_img_dir, src_lbl_dir, dst_root, split):
    copied=0
    for img in list_images(src_img_dir):
        lbl=src_lbl_dir/(img.stem+'.txt')
        if not lbl.exists():
            raise RuntimeError(f'Missing real label for {img}')
        shutil.copy2(img, dst_root/'images'/split/img.name)
        shutil.copy2(lbl, dst_root/'labels'/split/lbl.name)
        copied += 1
    return copied


def copy_aug_train(aug_root, prefix, dst_root):
    copied=0
    for img in list_images(aug_root/'images'):
        lbl=aug_root/'labels'/(img.stem+'.txt')
        if not lbl.exists():
            raise RuntimeError(f'Missing aug label for {img}')
        shutil.copy2(img, dst_root/'images'/'train'/(prefix+img.name))
        shutil.copy2(lbl, dst_root/'labels'/'train'/(prefix+lbl.name))
        copied += 1
    return copied


def write_data_yaml(ds_root):
    lines = [
        f'path: {ds_root}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        '',
        'names:',
        '  0: D00_longitudinal_crack',
        '  1: D10_transverse_crack',
        '  2: D20_alligator_crack',
        '  3: D40_pothole',
    ]
    (ds_root/'data.yaml').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def analyze_split(ds_root, split, aug_prefix):
    img_dir=ds_root/'images'/split
    lbl_dir=ds_root/'labels'/split
    imgs=list_images(img_dir)
    labels=sorted([p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower()=='.txt'])
    img_stems={p.stem for p in imgs}
    lbl_stems={p.stem for p in labels}
    pair_match=img_stems==lbl_stems
    errors=[]
    if not pair_match:
        errors.append(f'{split}: image/label stems mismatch')
    bbox_count=0
    class_counts=Counter()
    empty_real=[]
    empty_aug=[]
    bad=[]
    for lbl in labels:
        text=lbl.read_text(encoding='utf-8', errors='ignore').strip()
        is_aug = lbl.name.startswith(aug_prefix)
        if not text:
            if is_aug:
                empty_aug.append(lbl.name)
            else:
                empty_real.append(lbl.name)
            continue
        for ln,line in enumerate(text.splitlines(),1):
            parts=line.split()
            if len(parts)<5:
                errors.append(f'{split}: bad label line {lbl.name}:{ln}')
                continue
            try:
                cid=int(float(parts[0]))
            except Exception:
                errors.append(f'{split}: invalid class id {lbl.name}:{ln}')
                continue
            if cid not in CLASS_NAMES:
                bad.append(f'{lbl.name}:{ln}:{cid}')
            else:
                bbox_count += 1
                class_counts[cid] += 1
    if empty_aug:
        errors.append(f'{split}: empty generated labels count={len(empty_aug)} examples={empty_aug[:5]}')
    if bad:
        errors.append(f'{split}: invalid class ids count={len(bad)} examples={bad[:5]}')
    return {
        'image_count':len(imgs), 'label_count':len(labels), 'bbox_count':bbox_count,
        'D00_bbox':class_counts[0], 'D10_bbox':class_counts[1], 'D20_bbox':class_counts[2], 'D40_bbox':class_counts[3],
        'pair_match':pair_match, 'empty_real_label_count':len(empty_real), 'empty_aug_label_count':len(empty_aug), 'errors':errors,
    }


def validate_aug_only_train(ds_root, prefix):
    errors=[]
    for split in ['val','test']:
        aug_imgs=[p.name for p in list_images(ds_root/'images'/split) if p.name.startswith(prefix)]
        aug_lbls=[p.name for p in (ds_root/'labels'/split).glob('*.txt') if p.name.startswith(prefix)]
        if aug_imgs or aug_lbls:
            errors.append(f'{split}: generated files with prefix {prefix} found images={len(aug_imgs)} labels={len(aug_lbls)}')
    return errors


def main():
    layout, real = detect_real_layout(REAL_ROOT)
    print(f'real_layout={layout}', flush=True)
    real_counts={split: len(list_images(real[split]['images'])) for split in ['train','val','test']}
    print(f'real_counts={real_counts}', flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows=[]
    failures=[]
    for ds_name,cfg in AUG_SETS.items():
        print(f'Building {ds_name}', flush=True)
        ds_root=OUT_ROOT/ds_name
        reset_dataset_dir(ds_root)
        for split in ['train','val','test']:
            copy_real_split(real[split]['images'], real[split]['labels'], ds_root, split)
        aug_count=copy_aug_train(cfg['root'], cfg['prefix'], ds_root)
        if aug_count != 200:
            failures.append(f'{ds_name}: aug_count {aug_count} != 200')
        write_data_yaml(ds_root)
        ds_errors=[]
        split_stats={}
        for split in ['train','val','test']:
            st=analyze_split(ds_root, split, cfg['prefix'])
            split_stats[split]=st
            ds_errors.extend(st['errors'])
            all_rows.append({
                'dataset_name':ds_name, 'split':split,
                'image_count':st['image_count'], 'label_count':st['label_count'], 'bbox_count':st['bbox_count'],
                'D00_bbox':st['D00_bbox'], 'D10_bbox':st['D10_bbox'], 'D20_bbox':st['D20_bbox'], 'D40_bbox':st['D40_bbox'],
                'pair_match':st['pair_match'], 'empty_real_label_count':st['empty_real_label_count'], 'empty_aug_label_count':st['empty_aug_label_count'],
                'data_yaml_path':str(ds_root/'data.yaml'),
            })
        ds_errors.extend(validate_aug_only_train(ds_root, cfg['prefix']))
        if split_stats['val']['image_count'] != real_counts['val'] or split_stats['val']['label_count'] != real_counts['val']:
            ds_errors.append(f'val count changed from real {real_counts["val"]}')
        if split_stats['test']['image_count'] != real_counts['test'] or split_stats['test']['label_count'] != real_counts['test']:
            ds_errors.append(f'test count changed from real {real_counts["test"]}')
        ready=len(ds_errors)==0
        (ds_root/'ready_for_training.txt').write_text(f'ready_for_training = {ready}\n', encoding='utf-8')
        pd.DataFrame([r for r in all_rows if r['dataset_name']==ds_name]).drop(columns=['data_yaml_path']).to_csv(ds_root/'dataset_class_distribution.csv', index=False)
        lines=[f'===== {ds_name} Dataset Summary =====', f'Real root: {REAL_ROOT}', f'Real layout: {layout}', f'Aug root: {cfg["root"]}', f'Aug prefix: {cfg["prefix"]}', f'Aug train images added: {aug_count}', f'Data YAML: {ds_root/"data.yaml"}', '']
        for split in ['train','val','test']:
            st=split_stats[split]
            lines.append(f'{split}: images={st["image_count"]}, labels={st["label_count"]}, pair_match={st["pair_match"]}, bbox={st["bbox_count"]}, D00={st["D00_bbox"]}, D10={st["D10_bbox"]}, D20={st["D20_bbox"]}, D40={st["D40_bbox"]}, empty_real_labels={st["empty_real_label_count"]}, empty_aug_labels={st["empty_aug_label_count"]}')
        lines += ['', f'ready_for_training = {ready}']
        if ds_errors:
            lines += ['Errors:', *ds_errors]
            failures.extend([f'{ds_name}: {e}' for e in ds_errors])
        (ds_root/'dataset_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines), flush=True)
    summary_df=pd.DataFrame(all_rows)
    summary_df.to_csv(OUT_ROOT/'ch4_dataset_summary.csv', index=False)
    if failures:
        (OUT_ROOT/'ch4_dataset_build_errors.txt').write_text('\n'.join(failures)+'\n', encoding='utf-8')
        raise RuntimeError('Dataset build validation failed:\n'+'\n'.join(failures[:30]))
    print('All datasets ready_for_training = True', flush=True)
    print(summary_df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
