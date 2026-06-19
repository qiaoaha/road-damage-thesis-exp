
from pathlib import Path
import shutil
import tarfile
import pandas as pd
from collections import Counter

ROOT = Path('/root/autodl-tmp/road_damage_exp')
REAL_ROOT = ROOT / 'processed/real_yolo'
BASE80_ROOT = ROOT / 'processed/base80_yolo'
BASE80_TAR = ROOT / 'base80_upload.tar.gz'
OUT_ROOT = ROOT / 'datasets_yolo_ch4_base80'
BASE80_ONLY = OUT_ROOT / 'base80_only'
DATASETS = ['base80_only', 'base80_plus_random_200', 'base80_plus_lpips_200', 'base80_plus_ours_200']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_DIR_TO_PREFIX = {'D00_longitudinal_crack':'D00','D10_transverse_crack':'D10','D20_alligator_crack':'D20','D40_pothole':'D40'}
CLASS_NAMES = {0:'D00_longitudinal_crack',1:'D10_transverse_crack',2:'D20_alligator_crack',3:'D40_pothole'}


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
    mapping={}
    with tarfile.open(BASE80_TAR, 'r:gz') as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            path=Path(m.name)
            if path.suffix.lower() not in IMG_EXTS:
                continue
            for clsdir,prefix in CLASS_DIR_TO_PREFIX.items():
                if clsdir in path.parts:
                    mapping[path.stem]=prefix
                    break
    return mapping


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


def copy_base80_train(dst_root):
    target_map=load_base80_target_map()
    pairs=[]
    for img in list_images(BASE80_ROOT/'images/train'):
        lbl=BASE80_ROOT/'labels/train'/(img.stem+'.txt')
        target=target_map.get(img.stem)
        if not lbl.exists() or target is None:
            raise RuntimeError(f'Base80 match failed: {img} label_exists={lbl.exists()} target={target}')
        pairs.append((img,lbl,target))
    counts=Counter(t for _,_,t in pairs)
    if len(pairs)!=80 or any(counts.get(k,0)!=20 for k in ['D00','D10','D20','D40']):
        raise RuntimeError(f'Invalid base80 target counts: total={len(pairs)} counts={dict(counts)}')
    for img,lbl,target in pairs:
        shutil.copy2(img, dst_root/'images/train'/('base80_'+img.name))
        shutil.copy2(lbl, dst_root/'labels/train'/('base80_'+lbl.name))
    return counts


def copy_real_split(real, split, dst_root):
    for img in list_images(real[split]['images']):
        lbl=real[split]['labels']/(img.stem+'.txt')
        if not lbl.exists():
            raise RuntimeError(f'Missing real label: {img}')
        shutil.copy2(img, dst_root/'images'/split/img.name)
        shutil.copy2(lbl, dst_root/'labels'/split/lbl.name)


def write_data_yaml(ds_root):
    text='\n'.join([
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
    ])+'\n'
    (ds_root/'data.yaml').write_text(text, encoding='utf-8')


def write_data_yolov5(ds_root):
    text='\n'.join([
        f'path: {ds_root}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        '',
        'nc: 4',
        "names: ['D00_longitudinal_crack', 'D10_transverse_crack', 'D20_alligator_crack', 'D40_pothole']",
    ])+'\n'
    (ds_root/'data_yolov5.yaml').write_text(text, encoding='utf-8')


def analyze_split(ds_root, split):
    imgs=list_images(ds_root/'images'/split)
    lbls=sorted((ds_root/'labels'/split).glob('*.txt'))
    pair_match={p.stem for p in imgs} == {p.stem for p in lbls}
    empty=0; bbox=0; cls=Counter(); bad=[]
    for lbl in lbls:
        text=lbl.read_text(encoding='utf-8', errors='ignore').strip()
        if not text:
            empty += 1; continue
        for ln,line in enumerate(text.splitlines(),1):
            parts=line.split()
            try: cid=int(float(parts[0]))
            except Exception:
                bad.append(f'{lbl.name}:{ln}:bad'); continue
            if cid not in CLASS_NAMES:
                bad.append(f'{lbl.name}:{ln}:{cid}')
            else:
                bbox += 1; cls[cid] += 1
    return {'image_count':len(imgs),'label_count':len(lbls),'empty_label_count':empty,'bbox_count':bbox,'D00_bbox':cls[0],'D10_bbox':cls[1],'D20_bbox':cls[2],'D40_bbox':cls[3],'base80_image_count':sum(1 for p in imgs if p.name.startswith('base80_')) if split=='train' else 0,'generated_image_count':sum(1 for p in imgs if p.name.startswith(('random_','lpips_','ours_'))) if split=='train' else 0,'pair_match':pair_match,'bad':bad}


def summarize_dataset(ds_root, ds_name, base_counts=None):
    rows=[]; ready=True; errors=[]
    for split in ['train','val','test']:
        st=analyze_split(ds_root, split)
        if not st['pair_match'] or st['bad']:
            ready=False; errors.extend(st['bad'][:10])
        if ds_name == 'base80_only':
            expected={'train':80,'val':296,'test':298}[split]
        else:
            expected={'train':280,'val':296,'test':298}[split]
        if st['image_count']!=expected or st['label_count']!=expected:
            ready=False; errors.append(f'{split} count {st["image_count"]}/{st["label_count"]} expected {expected}')
        rows.append({'dataset_name':ds_name,'split':split,**{k:v for k,v in st.items() if k!='bad'},'data_yaml_path':str(ds_root/'data.yaml'),'data_yolov5_yaml_path':str(ds_root/'data_yolov5.yaml'),'ready_for_training':ready})
    if ds_name == 'base80_only':
        train=rows[0]
        if train['base80_image_count']!=80 or train['generated_image_count']!=0:
            ready=False; errors.append('base80_only train composition invalid')
    (ds_root/'ready_for_training.txt').write_text(f'ready_for_training = {ready}\n', encoding='utf-8')
    for r in rows:
        r['ready_for_training']=ready
    pd.DataFrame(rows).drop(columns=['data_yaml_path','data_yolov5_yaml_path']).to_csv(ds_root/'dataset_class_distribution.csv', index=False)
    lines=[f'===== {ds_name} Dataset Summary =====',f'Base80 source: {BASE80_ROOT}',f'Base80 class source: {BASE80_TAR}',f'Data YAML: {ds_root/"data.yaml"}',f'YOLOv5 YAML: {ds_root/"data_yolov5.yaml"}','']
    if base_counts:
        lines.append(f'base80_target_class_counts: {dict(base_counts)}')
    for r in rows:
        lines.append(f'{r["split"]}: images={r["image_count"]}, labels={r["label_count"]}, empty_labels={r["empty_label_count"]}, pair_match={r["pair_match"]}, bbox={r["bbox_count"]}, D00={r["D00_bbox"]}, D10={r["D10_bbox"]}, D20={r["D20_bbox"]}, D40={r["D40_bbox"]}, base80_images={r["base80_image_count"]}, generated_images={r["generated_image_count"]}')
    lines += ['', f'ready_for_training = {ready}']
    if errors:
        lines += ['Errors:', *errors]
    (ds_root/'dataset_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if not ready:
        raise RuntimeError(f'{ds_name} not ready: {errors}')
    return rows


def main():
    layout, real = detect_real_layout(REAL_ROOT)
    reset_dataset_dir(BASE80_ONLY)
    base_counts=copy_base80_train(BASE80_ONLY)
    for split in ['val','test']:
        copy_real_split(real, split, BASE80_ONLY)
    write_data_yaml(BASE80_ONLY)
    write_data_yolov5(BASE80_ONLY)
    all_rows=summarize_dataset(BASE80_ONLY, 'base80_only', base_counts)
    for ds in DATASETS[1:]:
        ds_root=OUT_ROOT/ds
        if not (ds_root/'data.yaml').exists():
            raise RuntimeError(f'Missing existing dataset: {ds_root}')
        write_data_yolov5(ds_root)
        all_rows.extend(summarize_dataset(ds_root, ds))
    pd.DataFrame(all_rows).to_csv(OUT_ROOT/'ch4_base80_dataset_summary_with_base80_only.csv', index=False)
    print('base80_only built and data_yolov5.yaml generated for all datasets')
    print(pd.DataFrame(all_rows).to_string(index=False))

if __name__ == '__main__':
    main()
