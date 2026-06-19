
from pathlib import Path
import shutil
import pandas as pd
import tarfile
from collections import Counter, defaultdict

REAL_ROOT = Path('/root/autodl-tmp/road_damage_exp/processed/real_yolo')
BASE80_ROOT = Path('/root/autodl-tmp/road_damage_exp/processed/base80_yolo')
OUT_ROOT = Path('/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80')
AUG_SETS = {
    'base80_plus_random_200': {'root': Path('/root/autodl-tmp/road_damage_exp/screening/random_200'), 'prefix': 'random_'},
    'base80_plus_lpips_200': {'root': Path('/root/autodl-tmp/road_damage_exp/screening/lpips_200'), 'prefix': 'lpips_'},
    'base80_plus_ours_200': {'root': Path('/root/autodl-tmp/road_damage_exp/screening/final_ours_200'), 'prefix': 'ours_'},
}
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_NAMES = {0:'D00_longitudinal_crack',1:'D10_transverse_crack',2:'D20_alligator_crack',3:'D40_pothole'}
CLASS_PREFIX_BY_ID = {0:'D00', 1:'D10', 2:'D20', 3:'D40'}
BASE80_TAR = Path('/root/autodl-tmp/road_damage_exp/base80_upload.tar.gz')
CLASS_DIR_TO_PREFIX = {'D00_longitudinal_crack':'D00','D10_transverse_crack':'D10','D20_alligator_crack':'D20','D40_pothole':'D40'}


def list_images(d):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]) if d.exists() else []


def detect_real_layout(root):
    fmt_a = all((root/split/sub).exists() for split in ['train','val','test'] for sub in ['images','labels'])
    fmt_b = all((root/sub/split).exists() for split in ['train','val','test'] for sub in ['images','labels'])
    if fmt_a:
        return 'A', {split:{'images':root/split/'images','labels':root/split/'labels'} for split in ['train','val','test']}
    if fmt_b:
        return 'B', {split:{'images':root/'images'/split,'labels':root/'labels'/split} for split in ['train','val','test']}
    raise RuntimeError(f'Cannot detect real_yolo layout under {root}')



def load_base80_target_map():
    mapping = {}
    if BASE80_TAR.exists():
        with tarfile.open(BASE80_TAR, 'r:gz') as tar:
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                path = Path(m.name)
                if path.suffix.lower() not in IMG_EXTS:
                    continue
                parts = path.parts
                for class_dir, prefix in CLASS_DIR_TO_PREFIX.items():
                    if class_dir in parts:
                        mapping[path.stem] = prefix
                        break
    return mapping


def detect_base80(root, real_paths):
    target_map = load_base80_target_map()
    candidates = []
    if (root/'images'/'train').exists():
        img_dir = root/'images'/'train'
        lbl_dir = root/'labels'/'train'
        for img in list_images(img_dir):
            candidates.append((img, lbl_dir/(img.stem+'.txt'), target_map.get(img.stem)))
    elif (root/'train'/'images').exists():
        img_dir = root/'train'/'images'
        lbl_dir = root/'train'/'labels'
        for img in list_images(img_dir):
            candidates.append((img, lbl_dir/(img.stem+'.txt'), target_map.get(img.stem)))
    elif any((root/c).exists() for c in ['D00','D10','D20','D40']):
        label_index = {}
        for split in ['train','val','test']:
            for lbl in real_paths[split]['labels'].glob('*.txt'):
                label_index[lbl.stem] = lbl
        for cls in ['D00','D10','D20','D40']:
            for img in list_images(root/cls):
                candidates.append((img, label_index.get(img.stem), cls))
    else:
        label_index = {}
        for lbl in root.rglob('*.txt'):
            label_index[lbl.stem] = lbl
        for split in ['train','val','test']:
            for lbl in real_paths[split]['labels'].glob('*.txt'):
                label_index.setdefault(lbl.stem, lbl)
        for img in list_images(root):
            candidates.append((img, label_index.get(img.stem), target_map.get(img.stem)))
    unmatched = [str(img) for img,lbl,target in candidates if lbl is None or not Path(lbl).exists()]
    missing_target = [str(img) for img,lbl,target in candidates if target is None]
    if unmatched or missing_target:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        if unmatched:
            (OUT_ROOT/'unmatched_base80_images.txt').write_text('\n'.join(unmatched)+'\n', encoding='utf-8')
        if missing_target:
            (OUT_ROOT/'missing_base80_target_class.txt').write_text('\n'.join(missing_target)+'\n', encoding='utf-8')
        raise RuntimeError(f'Base80 matching failed: labels_missing={len(unmatched)}, target_missing={len(missing_target)}')
    return [(img, Path(lbl), target) for img,lbl,target in candidates]

def first_class_id(label_path):
    text = label_path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return None
    for line in text.splitlines():
        parts=line.split()
        if parts:
            cid=int(float(parts[0]))
            if cid in CLASS_NAMES:
                return cid
    return None


def classify_aug_by_name(img_name):
    for prefix in ['D00','D10','D20','D40']:
        if img_name.startswith(prefix + '_'):
            return prefix
    return 'UNKNOWN'


def reset_dataset_dir(root):
    if root.exists():
        for p in sorted(root.rglob('*'), reverse=True):
            if p.is_file(): p.unlink()
            elif p.is_dir():
                try: p.rmdir()
                except OSError: pass
    for split in ['train','val','test']:
        (root/'images'/split).mkdir(parents=True, exist_ok=True)
        (root/'labels'/split).mkdir(parents=True, exist_ok=True)


def copy_real_split(src_img_dir, src_lbl_dir, dst_root, split):
    c=0
    for img in list_images(src_img_dir):
        lbl=src_lbl_dir/(img.stem+'.txt')
        if not lbl.exists(): raise RuntimeError(f'Missing real label: {img}')
        shutil.copy2(img, dst_root/'images'/split/img.name)
        shutil.copy2(lbl, dst_root/'labels'/split/lbl.name)
        c += 1
    return c


def copy_base80_train(base_pairs, dst_root):
    class_counts=Counter()
    for img,lbl,target in base_pairs:
        if target not in ['D00','D10','D20','D40']:
            raise RuntimeError(f'Base80 missing target class: {img}')
        class_counts[target] += 1
        shutil.copy2(img, dst_root/'images'/'train'/('base80_'+img.name))
        shutil.copy2(lbl, dst_root/'labels'/'train'/('base80_'+lbl.name))
    return class_counts


def copy_aug_train(aug_root, prefix, dst_root):
    class_counts=Counter()
    c=0
    for img in list_images(aug_root/'images'):
        lbl=aug_root/'labels'/(img.stem+'.txt')
        if not lbl.exists(): raise RuntimeError(f'Missing aug label: {img}')
        cls=classify_aug_by_name(img.name)
        class_counts[cls] += 1
        shutil.copy2(img, dst_root/'images'/'train'/(prefix+img.name))
        shutil.copy2(lbl, dst_root/'labels'/'train'/(prefix+lbl.name))
        c += 1
    return c, class_counts


def write_data_yaml(ds_root):
    lines=[f'path: {ds_root}','train: images/train','val: images/val','test: images/test','','names:','  0: D00_longitudinal_crack','  1: D10_transverse_crack','  2: D20_alligator_crack','  3: D40_pothole']
    (ds_root/'data.yaml').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def analyze_split(ds_root, split):
    imgs=list_images(ds_root/'images'/split)
    labels=sorted([p for p in (ds_root/'labels'/split).iterdir() if p.is_file() and p.suffix.lower()=='.txt'])
    img_stems={p.stem for p in imgs}; lbl_stems={p.stem for p in labels}
    pair_match=img_stems==lbl_stems
    errors=[]
    if not pair_match: errors.append(f'{split}: image/label stems mismatch')
    class_counts=Counter(); bbox=0; empty=0; bad=[]
    base80_count=sum(1 for p in imgs if p.name.startswith('base80_')) if split=='train' else 0
    gen_count=sum(1 for p in imgs if p.name.startswith(('random_','lpips_','ours_'))) if split=='train' else 0
    for lbl in labels:
        text=lbl.read_text(encoding='utf-8', errors='ignore').strip()
        if not text:
            empty += 1; continue
        for ln,line in enumerate(text.splitlines(),1):
            parts=line.split()
            if len(parts)<5:
                errors.append(f'{split}: bad label line {lbl.name}:{ln}'); continue
            try: cid=int(float(parts[0]))
            except Exception:
                errors.append(f'{split}: invalid class id {lbl.name}:{ln}'); continue
            if cid not in CLASS_NAMES: bad.append(f'{lbl.name}:{ln}:{cid}')
            else: bbox += 1; class_counts[cid] += 1
    if bad: errors.append(f'{split}: invalid class ids count={len(bad)} examples={bad[:5]}')
    return {'image_count':len(imgs),'label_count':len(labels),'empty_label_count':empty,'bbox_count':bbox,'D00_bbox':class_counts[0],'D10_bbox':class_counts[1],'D20_bbox':class_counts[2],'D40_bbox':class_counts[3],'base80_image_count':base80_count,'generated_image_count':gen_count,'pair_match':pair_match,'errors':errors}


def validate_aug_only_train(ds_root):
    errors=[]
    for split in ['val','test']:
        imgs=list_images(ds_root/'images'/split)
        bad=[p.name for p in imgs if p.name.startswith(('base80_','random_','lpips_','ours_'))]
        if bad: errors.append(f'{split}: non-real prefixed files found {bad[:5]}')
    return errors


def main():
    layout, real = detect_real_layout(REAL_ROOT)
    base_pairs = detect_base80(BASE80_ROOT, real)
    base_counts = Counter(target for img,lbl,target in base_pairs)
    print('real_layout=', layout, flush=True)
    print('base80_source=', BASE80_ROOT, flush=True)
    print('base80_count=', len(base_pairs), 'base80_class_counts=', dict(base_counts), flush=True)
    expected_base={'D00':20,'D10':20,'D20':20,'D40':20}
    failures=[]
    if len(base_pairs)!=80: failures.append(f'base80 image count {len(base_pairs)} != 80')
    for k,v in expected_base.items():
        if base_counts.get(k,0)!=v: failures.append(f'base80 {k} count {base_counts.get(k,0)} != {v}')
    real_counts={split:len(list_images(real[split]['images'])) for split in ['train','val','test']}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows=[]
    for ds_name,cfg in AUG_SETS.items():
        ds_root=OUT_ROOT/ds_name
        reset_dataset_dir(ds_root)
        base_c=copy_base80_train(base_pairs, ds_root)
        aug_total, aug_c=copy_aug_train(cfg['root'], cfg['prefix'], ds_root)
        for split in ['val','test']:
            copy_real_split(real[split]['images'], real[split]['labels'], ds_root, split)
        write_data_yaml(ds_root)
        ds_errors=[]; split_stats={}
        for split in ['train','val','test']:
            st=analyze_split(ds_root, split); split_stats[split]=st; ds_errors.extend(st['errors'])
            all_rows.append({'dataset_name':ds_name,'split':split,'image_count':st['image_count'],'label_count':st['label_count'],'empty_label_count':st['empty_label_count'],'bbox_count':st['bbox_count'],'D00_bbox':st['D00_bbox'],'D10_bbox':st['D10_bbox'],'D20_bbox':st['D20_bbox'],'D40_bbox':st['D40_bbox'],'base80_image_count':st['base80_image_count'],'generated_image_count':st['generated_image_count'],'pair_match':st['pair_match'],'data_yaml_path':str(ds_root/'data.yaml'),'ready_for_training':False})
        ds_errors.extend(validate_aug_only_train(ds_root))
        if split_stats['train']['image_count']!=280 or split_stats['train']['label_count']!=280: ds_errors.append('train count must be 280/280')
        if split_stats['val']['image_count']!=296 or split_stats['val']['label_count']!=296: ds_errors.append('val count must be 296/296')
        if split_stats['test']['image_count']!=298 or split_stats['test']['label_count']!=298: ds_errors.append('test count must be 298/298')
        if split_stats['train']['base80_image_count']!=80: ds_errors.append('base80 train count must be 80')
        if split_stats['train']['generated_image_count']!=200: ds_errors.append('generated train count must be 200')
        for k,v in expected_base.items():
            if base_c.get(k,0)!=v: ds_errors.append(f'base80 {k} count {base_c.get(k,0)} != {v}')
            if aug_c.get(k,0)!=50: ds_errors.append(f'generated {k} count {aug_c.get(k,0)} != 50')
        if real_counts['val']!=296 or real_counts['test']!=298: ds_errors.append(f'real val/test unexpected {real_counts}')
        ready=not ds_errors
        for r in all_rows:
            if r['dataset_name']==ds_name: r['ready_for_training']=ready
        (ds_root/'ready_for_training.txt').write_text(f'ready_for_training = {ready}\n', encoding='utf-8')
        pd.DataFrame([r for r in all_rows if r['dataset_name']==ds_name]).drop(columns=['data_yaml_path']).to_csv(ds_root/'dataset_class_distribution.csv', index=False)
        lines=[f'===== {ds_name} Dataset Summary =====',f'Base80 source: {BASE80_ROOT}',f'Real root: {REAL_ROOT}',f'Real layout: {layout}',f'Aug root: {cfg["root"]}',f'Data YAML: {ds_root/"data.yaml"}','',f'base80_class_counts: {dict(base_c)}',f'generated_class_counts: {dict(aug_c)}','']
        for split in ['train','val','test']:
            st=split_stats[split]
            lines.append(f'{split}: images={st["image_count"]}, labels={st["label_count"]}, empty_labels={st["empty_label_count"]}, pair_match={st["pair_match"]}, bbox={st["bbox_count"]}, D00={st["D00_bbox"]}, D10={st["D10_bbox"]}, D20={st["D20_bbox"]}, D40={st["D40_bbox"]}, base80_images={st["base80_image_count"]}, generated_images={st["generated_image_count"]}')
        lines += ['', f'ready_for_training = {ready}']
        if ds_errors:
            lines += ['Errors:', *ds_errors]; failures.extend([f'{ds_name}: {e}' for e in ds_errors])
        (ds_root/'dataset_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines), flush=True)
    if failures:
        (OUT_ROOT/'ch4_base80_dataset_build_errors.txt').write_text('\n'.join(failures)+'\n', encoding='utf-8')
        raise RuntimeError('Validation failed:\n'+'\n'.join(failures[:50]))
    summary=pd.DataFrame(all_rows)
    summary.to_csv(OUT_ROOT/'ch4_base80_dataset_summary.csv', index=False)
    print('All base80 datasets ready_for_training = True', flush=True)
    print(summary.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
