#!/usr/bin/env python3
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

BASE = Path('/root/autodl-tmp/road_damage_exp')
RAW_ROOT = BASE / 'cross_domain'
OUT_ROOT = BASE / 'datasets_cross_domain'
SUMMARY = OUT_ROOT / 'cross_domain_dataset_summary.csv'

CLASSES = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}
DOMAINS = {
    'Japan': {'raw': RAW_ROOT / 'Japan_eval_500_raw', 'out': OUT_ROOT / 'japan_yolo'},
    'Norway': {'raw': RAW_ROOT / 'Norway_eval_500_raw', 'out': OUT_ROOT / 'norway_yolo'},
}
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']


def image_size(path: Path):
    with Image.open(path) as im:
        return im.size


def find_image(images_dir: Path, filename: str, stem: str):
    if filename:
        p = images_dir / filename
        if p.exists():
            return p
        lower = filename.lower()
        for q in images_dir.iterdir():
            if q.is_file() and q.name.lower() == lower:
                return q
    for ext in IMG_EXTS:
        p = images_dir / f'{stem}{ext}'
        if p.exists():
            return p
    for q in images_dir.iterdir():
        if q.is_file() and q.stem == stem and q.suffix.lower() in IMG_EXTS:
            return q
    return None


def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext('filename') or '').strip()
    size = root.find('size')
    w = int(float(size.findtext('width'))) if size is not None and size.findtext('width') else 0
    h = int(float(size.findtext('height'))) if size is not None and size.findtext('height') else 0
    boxes = []
    filtered_other = 0
    invalid = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        if name not in CLASSES:
            filtered_other += 1
            continue
        b = obj.find('bndbox')
        if b is None:
            invalid.append((name, 'missing_bndbox'))
            continue
        xmin = float(b.findtext('xmin'))
        ymin = float(b.findtext('ymin'))
        xmax = float(b.findtext('xmax'))
        ymax = float(b.findtext('ymax'))
        xmin = max(0.0, min(xmin, float(w)))
        xmax = max(0.0, min(xmax, float(w)))
        ymin = max(0.0, min(ymin, float(h)))
        ymax = max(0.0, min(ymax, float(h)))
        bw = xmax - xmin
        bh = ymax - ymin
        if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
            invalid.append((name, f'invalid_box:{xmin},{ymin},{xmax},{ymax},size={w}x{h}'))
            continue
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        nw = bw / w
        nh = bh / h
        vals = [xc, yc, nw, nh]
        if not all(0 <= v <= 1 for v in vals) or nw <= 0 or nh <= 0:
            invalid.append((name, f'normalized_invalid:{vals}'))
            continue
        boxes.append((CLASSES[name], xc, yc, nw, nh))
    return filename, w, h, boxes, filtered_other, invalid


def write_yaml(out_dir: Path):
    text = (
        f'path: {out_dir}\n'
        'train: images/test\n'
        'val: images/test\n'
        'test: images/test\n\n'
        'names:\n'
        '  0: D00_longitudinal_crack\n'
        '  1: D10_transverse_crack\n'
        '  2: D20_alligator_crack\n'
        '  3: D40_pothole\n'
    )
    (out_dir / 'data.yaml').write_text(text, encoding='utf-8')


def validate_yolo(out_dir: Path):
    img_dir = out_dir / 'images/test'
    lab_dir = out_dir / 'labels/test'
    images = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    labels = sorted([p for p in lab_dir.iterdir() if p.is_file() and p.suffix.lower() == '.txt'])
    img_stems = {p.stem for p in images}
    lab_stems = {p.stem for p in labels}
    pair_match = img_stems == lab_stems
    empty = 0
    bbox_count = 0
    cls_counts = {i: 0 for i in range(4)}
    bad = []
    for lp in labels:
        text = lp.read_text(encoding='utf-8').strip()
        if not text:
            empty += 1
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                bad.append(f'{lp}:{lineno}:bad_field_count')
                continue
            try:
                cid = int(parts[0]); vals = [float(x) for x in parts[1:]]
            except Exception:
                bad.append(f'{lp}:{lineno}:parse_error')
                continue
            if cid not in (0, 1, 2, 3):
                bad.append(f'{lp}:{lineno}:bad_class:{cid}')
            if not all(0 <= v <= 1 for v in vals):
                bad.append(f'{lp}:{lineno}:coord_out_of_range:{vals}')
            if vals[2] <= 0 or vals[3] <= 0:
                bad.append(f'{lp}:{lineno}:non_positive_wh:{vals}')
            bbox_count += 1
            if cid in cls_counts:
                cls_counts[cid] += 1
    return {
        'image_count': len(images),
        'label_count': len(labels),
        'pair_match': pair_match,
        'empty_label_count': empty,
        'bbox_count': bbox_count,
        'class_counts': cls_counts,
        'bad': bad,
    }


def prepare_domain(domain: str, raw: Path, out: Path):
    images_dir = raw / 'images'
    xml_dir = raw / 'annotations/xmls'
    if not images_dir.exists() or not xml_dir.exists():
        raise FileNotFoundError(f'Missing raw images/xmls for {domain}: {raw}')
    if out.exists():
        shutil.rmtree(out)
    img_out = out / 'images/test'
    lab_out = out / 'labels/test'
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    unmatched = []
    invalid_records = []
    filtered_other_total = 0
    for xml_path in sorted(xml_dir.glob('*.xml')):
        filename, w, h, boxes, filtered_other, invalid = parse_xml(xml_path)
        filtered_other_total += filtered_other
        stem = xml_path.stem
        img = find_image(images_dir, filename, stem)
        if img is None:
            unmatched.append(str(xml_path))
            continue
        if w <= 0 or h <= 0:
            w2, h2 = image_size(img)
            w, h = w2, h2
        shutil.copy2(img, img_out / img.name)
        label_name = f'{Path(img.name).stem}.txt'
        with (lab_out / label_name).open('w', encoding='utf-8') as f:
            for cid, xc, yc, nw, nh in boxes:
                f.write(f'{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n')
        for item in invalid:
            invalid_records.append({'xml': str(xml_path), 'issue': repr(item)})
    if unmatched:
        unmatched_path = out / 'unmatched_xmls.txt'
        unmatched_path.write_text('\n'.join(unmatched), encoding='utf-8')
        raise RuntimeError(f'{domain}: unmatched XML/image count={len(unmatched)}; see {unmatched_path}')
    if invalid_records:
        invalid_path = out / 'invalid_boxes.csv'
        with invalid_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['xml','issue'])
            writer.writeheader(); writer.writerows(invalid_records)
        raise RuntimeError(f'{domain}: invalid boxes count={len(invalid_records)}; see {invalid_path}')
    write_yaml(out)
    stat = validate_yolo(out)
    if stat['bad']:
        bad_path = out / 'label_validation_errors.txt'
        bad_path.write_text('\n'.join(stat['bad']), encoding='utf-8')
        raise RuntimeError(f'{domain}: YOLO label validation failed; see {bad_path}')
    return stat, filtered_other_total


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, cfg in DOMAINS.items():
        stat, filtered_other = prepare_domain(domain, cfg['raw'], cfg['out'])
        row = {
            'domain': domain,
            'raw_dir': str(cfg['raw']),
            'yolo_dir': str(cfg['out']),
            'data_yaml': str(cfg['out'] / 'data.yaml'),
            'image_count': stat['image_count'],
            'label_count': stat['label_count'],
            'empty_label_count': stat['empty_label_count'],
            'bbox_count': stat['bbox_count'],
            'D00_bbox': stat['class_counts'][0],
            'D10_bbox': stat['class_counts'][1],
            'D20_bbox': stat['class_counts'][2],
            'D40_bbox': stat['class_counts'][3],
            'filtered_other_bbox_count': filtered_other,
            'pair_match': stat['pair_match'],
            'valid_label_classes': True,
            'bbox_coords_valid': True,
        }
        rows.append(row)
        print(row)
    with SUMMARY.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f'wrote {SUMMARY}')

if __name__ == '__main__':
    main()
