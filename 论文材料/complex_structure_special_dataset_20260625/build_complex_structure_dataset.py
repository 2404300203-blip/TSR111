import csv
import json
import re
import shutil
from pathlib import Path
from collections import Counter

ROOT = Path('/root/autodl-tmp/PaddleOCR')
OUT = ROOT / '论文材料' / 'complex_structure_special_dataset_20260625'
IMG_OUT = OUT / 'images'
FIG_OUT = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

paths = {
    'baseline': ROOT / 'output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl',
    'phase5e': ROOT / 'output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl',
    'horizontal': ROOT / 'output/phase10_axis_ablation_20260624/expC_horizontal_sample_teds.jsonl',
    'vertical': ROOT / 'output/phase10_axis_ablation_20260624/expD_vertical_sample_teds.jsonl',
}

COL_RE = re.compile(r'colspan\s*=\s*["\']?(\d+)["\']?')
ROW_RE = re.compile(r'rowspan\s*=\s*["\']?(\d+)["\']?')

def load_jsonl(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[int(row['sample_index'])] = row
    return rows

def parse_span(parts):
    text = ''.join(str(p) for p in parts)
    rowspan = 1
    colspan = 1
    mr = ROW_RE.search(text)
    mc = COL_RE.search(text)
    if mr:
        rowspan = max(1, int(mr.group(1)))
    if mc:
        colspan = max(1, int(mc.group(1)))
    return rowspan, colspan, text

def collect_td_opening(tokens, start):
    parts = [str(tokens[start])]
    j = start + 1
    if '>' in parts[0]:
        return parts, start
    while j < len(tokens):
        tok = str(tokens[j])
        if tok.startswith('<td') or tok in {'<tr>', '</tr>', '<thead>', '</thead>', '<tbody>', '</tbody>'}:
            return parts, j - 1
        parts.append(tok)
        if '>' in tok:
            return parts, j
        j += 1
    return parts, j - 1

def analyze_tokens(tokens):
    rows = []
    section = 'body'
    cur = None
    header_rows = 0
    body_rows = 0
    span_tokens = 0
    rowspan_tokens = 0
    colspan_tokens = 0
    max_rowspan = 1
    max_colspan = 1
    span_area = 0
    plain_cells = 0
    explicit_cell_count = 0
    reconstructed_td_openings = []

    i = 0
    while i < len(tokens):
        tok = str(tokens[i])
        if tok == '<thead>':
            section = 'head'
        elif tok == '</thead>':
            section = 'body'
        elif tok == '<tbody>':
            section = 'body'
        elif tok == '<tr>':
            cur = {'section': section, 'width': 0, 'cells': 0, 'spans': 0, 'rowspans': 0, 'colspans': 0}
        elif tok == '</tr>':
            if cur is not None:
                rows.append(cur)
                if cur['section'] == 'head':
                    header_rows += 1
                else:
                    body_rows += 1
            cur = None
        elif cur is not None and tok.startswith('<td'):
            parts, end_idx = collect_td_opening(tokens, i)
            rr, cc, opening_text = parse_span(parts)
            explicit_cell_count += 1
            reconstructed_td_openings.append(opening_text)
            cur['width'] += cc
            cur['cells'] += 1
            if rr > 1 or cc > 1:
                cur['spans'] += 1
                span_tokens += 1
                span_area += rr * cc
            else:
                plain_cells += 1
            if rr > 1:
                cur['rowspans'] += 1
                rowspan_tokens += 1
            if cc > 1:
                cur['colspans'] += 1
                colspan_tokens += 1
            max_rowspan = max(max_rowspan, rr)
            max_colspan = max(max_colspan, cc)
            i = end_idx
        i += 1

    widths = [r['width'] for r in rows]
    row_count = len(rows)
    max_width = max(widths) if widths else 0
    mean_width = sum(widths) / len(widths) if widths else 0.0
    width_variance = sum((w - mean_width) ** 2 for w in widths) / len(widths) if widths else 0.0
    head_span_rows = sum(1 for r in rows if r['section'] == 'head' and r['spans'] > 0)
    body_span_rows = sum(1 for r in rows if r['section'] == 'body' and r['spans'] > 0)
    horizontal_span_rows = sum(1 for r in rows if r['colspans'] > 0)
    vertical_span_rows = sum(1 for r in rows if r['rowspans'] > 0)
    max_row_cells = max((r['cells'] for r in rows), default=0)
    min_row_cells = min((r['cells'] for r in rows), default=0)

    return {
        'row_count': row_count,
        'header_rows': header_rows,
        'body_rows': body_rows,
        'max_width': max_width,
        'mean_width': mean_width,
        'width_variance': width_variance,
        'max_row_cells': max_row_cells,
        'min_row_cells': min_row_cells,
        'span_tokens': span_tokens,
        'rowspan_tokens': rowspan_tokens,
        'colspan_tokens': colspan_tokens,
        'max_rowspan': max_rowspan,
        'max_colspan': max_colspan,
        'span_area': span_area,
        'plain_cells': plain_cells,
        'explicit_cell_count': explicit_cell_count,
        'head_span_rows': head_span_rows,
        'body_span_rows': body_span_rows,
        'horizontal_span_rows': horizontal_span_rows,
        'vertical_span_rows': vertical_span_rows,
        'gt_len': len(tokens),
        'td_opening_examples': reconstructed_td_openings[:8],
    }

def bucket_tags(m):
    tags = []
    if m['header_rows'] >= 2 or m['head_span_rows'] > 0:
        tags.append('multi_level_header')
    if m['rowspan_tokens'] > 0 or m['colspan_tokens'] > 0:
        tags.append('row_col_span')
    if m['row_count'] >= 10 or m['max_width'] >= 6 or m['gt_len'] >= 80 or m['span_area'] >= 12:
        tags.append('long_range_dependency')
    if m['rowspan_tokens'] > 0:
        tags.append('vertical_dependency')
    if m['colspan_tokens'] > 0:
        tags.append('horizontal_dependency')
    if m['header_rows'] >= 2 and (m['rowspan_tokens'] > 0 or m['colspan_tokens'] > 0):
        tags.append('hierarchical_span_header')
    if m['row_count'] >= 15:
        tags.append('tall_table')
    if m['max_width'] >= 8:
        tags.append('wide_table')
    return tags

def complexity_score(r):
    return (
        r['header_rows'] * 4.0 +
        r['span_tokens'] * 4.0 +
        r['rowspan_tokens'] * 2.0 +
        r['colspan_tokens'] * 2.0 +
        r['max_rowspan'] * 1.5 +
        r['max_colspan'] * 1.5 +
        min(r['row_count'], 35) * 0.7 +
        min(r['max_width'], 16) * 0.8 +
        min(r['gt_len'], 180) * 0.05 +
        min(r['span_area'], 60) * 0.2
    )

def safe_float(v):
    return None if v is None else float(v)

def build_records():
    data = {k: load_jsonl(v) for k, v in paths.items()}
    base = data['baseline']
    phase = data['phase5e']
    horiz = data['horizontal']
    vert = data['vertical']
    if len(base) != 5000 or len(phase) != 5000:
        raise RuntimeError(f'Expected 5000 baseline/phase rows, got {len(base)} and {len(phase)}')

    records = []
    for idx, brow in sorted(base.items()):
        prow = phase[idx]
        m = analyze_tokens(brow['gt_tokens'])
        tags = bucket_tags(m)
        if not tags:
            continue
        hrow = horiz.get(idx)
        vrow = vert.get(idx)
        h_teds = safe_float(hrow.get('teds')) if hrow else None
        v_teds = safe_float(vrow.get('teds')) if vrow else None
        full_teds = float(prow['teds'])
        base_teds = float(brow['teds'])
        branch_tag = []
        if h_teds is not None and v_teds is not None:
            if full_teds > max(h_teds, v_teds) + 1e-12:
                branch_tag.append('dual_branch_complementary')
            if h_teds > v_teds + 0.05:
                branch_tag.append('horizontal_branch_better')
            if v_teds > h_teds + 0.05:
                branch_tag.append('vertical_branch_better')
            if full_teds > h_teds + 0.05 and full_teds > v_teds + 0.05:
                branch_tag.append('full_mamba_strong_gain_vs_single_axis')
        rec = {
            'sample_index': idx,
            'filename': brow['filename'],
            'image_path': brow['image_path'],
            'tags': tags,
            'branch_tags': branch_tag,
            'baseline_teds': base_teds,
            'phase5e_teds': full_teds,
            'delta_full_vs_baseline': full_teds - base_teds,
            'baseline_exact': bool(brow.get('exact')),
            'phase5e_exact': bool(prow.get('exact')),
            'horizontal_teds': h_teds,
            'vertical_teds': v_teds,
            **m,
        }
        rec['complexity_score'] = complexity_score(rec)
        rec['paper_use'] = []
        if 'hierarchical_span_header' in tags:
            rec['paper_use'].append('multi-level header with span')
        if 'horizontal_dependency' in tags:
            rec['paper_use'].append('horizontal branch / colspan evidence')
        if 'vertical_dependency' in tags:
            rec['paper_use'].append('vertical branch / rowspan evidence')
        if 'long_range_dependency' in tags:
            rec['paper_use'].append('long-range relation evidence')
        if branch_tag:
            rec['paper_use'].append('axis ablation evidence')
        records.append(rec)
    records.sort(key=lambda r: r['complexity_score'], reverse=True)
    return records

def choose_curated(records, target=500):
    curated = []
    seen = set()
    tag_plan = [
        ('hierarchical_span_header', 100),
        ('row_col_span', 120),
        ('horizontal_dependency', 100),
        ('vertical_dependency', 100),
        ('multi_level_header', 100),
        ('long_range_dependency', 100),
        ('wide_table', 60),
        ('tall_table', 60),
    ]
    for tag, quota in tag_plan:
        bucket = [r for r in records if tag in r['tags']]
        bucket.sort(key=lambda r: (r['delta_full_vs_baseline'], r['complexity_score']), reverse=True)
        for r in bucket[:quota]:
            if len(curated) >= target:
                break
            if r['sample_index'] not in seen:
                curated.append(r)
                seen.add(r['sample_index'])
    branch_bucket = [r for r in records if r['branch_tags']]
    branch_bucket.sort(key=lambda r: (len(r['branch_tags']), r['complexity_score']), reverse=True)
    for r in branch_bucket[:80]:
        if len(curated) >= target:
            break
        if r['sample_index'] not in seen:
            curated.append(r)
            seen.add(r['sample_index'])
    regressions = [r for r in records if r['delta_full_vs_baseline'] < -0.1]
    regressions.sort(key=lambda r: r['delta_full_vs_baseline'])
    for r in regressions[:60]:
        if len(curated) >= target:
            break
        if r['sample_index'] not in seen:
            curated.append(r)
            seen.add(r['sample_index'])
    for r in records:
        if len(curated) >= target:
            break
        if r['sample_index'] not in seen:
            curated.append(r)
            seen.add(r['sample_index'])
    curated.sort(key=lambda r: r['complexity_score'], reverse=True)
    return curated

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def copy_images(curated):
    for old in IMG_OUT.iterdir():
        if old.is_file():
            old.unlink()
    copied = 0
    for r in curated:
        src = ROOT / r['image_path']
        if src.exists():
            dst = IMG_OUT / (str(r['sample_index']).zfill(5) + '_' + r['filename'])
            shutil.copy2(src, dst)
            r['copied_image'] = str(dst.relative_to(OUT))
            copied += 1
        else:
            r['copied_image'] = None
    return copied

def write_csv(curated):
    fields = [
        'sample_index','filename','tags','branch_tags','paper_use','baseline_teds','phase5e_teds',
        'delta_full_vs_baseline','horizontal_teds','vertical_teds','complexity_score','header_rows',
        'row_count','max_width','span_tokens','rowspan_tokens','colspan_tokens','max_rowspan',
        'max_colspan','span_area','gt_len','copied_image','image_path'
    ]
    with (OUT / 'complex_structure_curated_500_metadata.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in curated:
            row = {k: r.get(k) for k in fields}
            for k in ['tags', 'branch_tags', 'paper_use']:
                row[k] = ';'.join(row[k] or [])
            w.writerow(row)

def summarize(records, curated, copied):
    counter = Counter()
    branch_counter = Counter()
    cur_counter = Counter()
    for r in records:
        counter.update(r['tags'])
        branch_counter.update(r['branch_tags'])
    for r in curated:
        cur_counter.update(r['tags'])
    summary = {
        'date': '2026-06-25',
        'source_split': 'train_data/table/pubtabnet/phase10_val_5000.jsonl',
        'full_complex_count': len(records),
        'curated_count': len(curated),
        'tag_counts_full': dict(counter),
        'tag_counts_curated': dict(cur_counter),
        'branch_tag_counts_full': dict(branch_counter),
        'phase5e_better_in_full': sum(1 for r in records if r['delta_full_vs_baseline'] > 1e-12),
        'phase5e_worse_in_full': sum(1 for r in records if r['delta_full_vs_baseline'] < -1e-12),
        'phase5e_same_in_full': sum(1 for r in records if abs(r['delta_full_vs_baseline']) <= 1e-12),
        'copied_images': copied,
        'span_parser_note': 'rowspan/colspan are parsed across split HTML tokens, e.g. <td + rowspan="2" + >.',
    }
    (OUT / 'dataset_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary, counter, branch_counter, cur_counter

def write_readme(summary, counter, branch_counter):
    meaning = {
        'multi_level_header':'多级表头，或表头区域出现跨行/跨列。',
        'row_col_span':'存在 rowspan 或 colspan。',
        'long_range_dependency':'行数、列数、HTML 序列长度或跨度面积较大，适合分析长距离结构依赖。',
        'vertical_dependency':'存在 rowspan，可作为纵向关系建模证据。',
        'horizontal_dependency':'存在 colspan，可作为横向关系建模证据。',
        'hierarchical_span_header':'多级表头同时含跨行/跨列，是复杂金融表头的核心类型。',
        'tall_table':'行数较多。',
        'wide_table':'列数较多。',
    }
    md = []
    md.append('# 复杂结构特殊数据集 Complex Structure Special Dataset\n\n')
    md.append('Date: 2026-06-25\n\n')
    md.append('本数据集从 Phase10 的 5000 样本验证集和逐样本诊断结果中筛选，用于支撑“基于双分支 Mamba 的单元格结构关系建模方法”这一创新点。筛选不是人工凭感觉挑图，而是依据 GT HTML token、TEDS 逐样本结果以及 horizontal/vertical axis ablation 结果自动生成。\n\n')
    md.append('## 数据来源\n\n')
    md.append('```text\ntrain_data/table/pubtabnet/phase10_val_5000.jsonl\noutput/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl\noutput/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl\noutput/phase10_axis_ablation_20260624/expC_horizontal_sample_teds.jsonl\noutput/phase10_axis_ablation_20260624/expD_vertical_sample_teds.jsonl\n```\n\n')
    md.append('## 数据规模\n\n')
    md.append('| 子集 | 数量 | 说明 |\n|---|---:|---|\n')
    md.append(f"| full complex metadata | {summary['full_complex_count']} | 5000 验证集中所有命中复杂结构规则的样本 |\n")
    md.append(f"| curated image subset | {summary['curated_count']} | 已复制图片的代表性复杂结构样本 |\n")
    md.append(f"| copied images | {summary['copied_images']} | 位于 `images/` 目录 |\n\n")
    md.append('## 复杂结构标签统计 full set\n\n')
    md.append('| 标签 | 数量 | 论文含义 |\n|---|---:|---|\n')
    for tag, count in counter.most_common():
        md.append(f"| `{tag}` | {count} | {meaning.get(tag, '')} |\n")
    md.append('\n## Axis Ablation 证据标签\n\n')
    md.append('| 标签 | 数量 | 含义 |\n|---|---:|---|\n')
    branch_meaning = {
        'dual_branch_complementary': '双分支结果高于单独 horizontal 和 vertical 分支。',
        'horizontal_branch_better': '横向分支显著优于纵向分支，常与 colspan / 宽表有关。',
        'vertical_branch_better': '纵向分支显著优于横向分支，常与 rowspan / 长表有关。',
        'full_mamba_strong_gain_vs_single_axis': '完整双分支相对两个单分支均有明显优势。',
    }
    if branch_counter:
        for tag, count in branch_counter.most_common():
            md.append(f"| `{tag}` | {count} | {branch_meaning.get(tag, '')} |\n")
    else:
        md.append('| - | 0 | 当前 axis ablation 未形成显著分支标签。 |\n')
    md.append('\n## 与创新点三的对应关系\n\n')
    md.append('- `horizontal_dependency` / `colspan`：对应横向逻辑关系建模。\n')
    md.append('- `vertical_dependency` / `rowspan`：对应纵向逻辑关系建模。\n')
    md.append('- `multi_level_header` 与 `hierarchical_span_header`：对应复杂金融表格常见的层次化表头。\n')
    md.append('- `long_range_dependency`：对应跨多个行列的长距离结构依赖。\n')
    md.append('- `dual_branch_complementary` 等 axis ablation 标签：可用于说明单分支不足，双分支 Mamba 更适合复杂结构关系建模。\n\n')
    md.append('## 文件说明\n\n')
    md.append('| 文件 | 说明 |\n|---|---|\n')
    md.append('| `complex_structure_full_5000_metadata.jsonl` | 全部复杂结构样本元数据 |\n')
    md.append('| `complex_structure_curated_500_metadata.jsonl` | 代表性 500 样本元数据 |\n')
    md.append('| `complex_structure_curated_500_metadata.csv` | 便于人工筛选的 CSV 表 |\n')
    md.append('| `complex_structure_curated_500_metadata_with_images.jsonl` | 含复制后图片路径的 500 样本元数据 |\n')
    md.append('| `images/` | 500 个代表性复杂表格图片 |\n')
    md.append('| `figures/` | 自动导出的代表性拼图和单图 |\n')
    md.append('| `dataset_summary.json` | 统计摘要 |\n\n')
    md.append('## 论文表述建议\n\n')
    md.append('```text\n为验证模型对复杂结构关系的建模能力，我们从验证集中额外构建了一个复杂结构特殊子集，覆盖多级表头、跨行跨列单元格以及长距离结构依赖。该子集根据 HTML 结构 token 中的 thead 层级、跨 token 解析得到的 rowspan/colspan、行列规模和序列长度自动筛选，并结合 horizontal/vertical axis ablation 结果分析双分支 Mamba 的贡献。\n```\n')
    (OUT / 'README.md').write_text(''.join(md), encoding='utf-8')

def generate_figures(curated):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        (FIG_OUT / 'FIGURE_GENERATION_SKIPPED.txt').write_text(str(exc), encoding='utf-8')
        return []
    for old in FIG_OUT.iterdir():
        if old.is_file():
            old.unlink()
    font_paths = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
    font = ImageFont.truetype(font_paths[0], 18) if Path(font_paths[0]).exists() else ImageFont.load_default()
    small = ImageFont.truetype(font_paths[0], 14) if Path(font_paths[0]).exists() else ImageFont.load_default()
    title_font = ImageFont.truetype(font_paths[1], 22) if Path(font_paths[1]).exists() else font

    def label(r):
        short_tags = ','.join(r['tags'][:3])
        return f"#{r['sample_index']} d={r['delta_full_vs_baseline']:+.3f} B={r['baseline_teds']:.3f} F={r['phase5e_teds']:.3f}\nspan={r['span_tokens']} row={r['row_count']} col={r['max_width']} {short_tags}"

    categories = [
        ('multi_level_header', 'multi_level_header_contact_sheet.png'),
        ('hierarchical_span_header', 'hierarchical_span_header_contact_sheet.png'),
        ('horizontal_dependency', 'horizontal_colspan_contact_sheet.png'),
        ('vertical_dependency', 'vertical_rowspan_contact_sheet.png'),
        ('long_range_dependency', 'long_range_dependency_contact_sheet.png'),
    ]
    made = []
    for tag, fname in categories:
        rows = [r for r in curated if tag in r['tags'] and r.get('copied_image')]
        rows.sort(key=lambda r: (r['delta_full_vs_baseline'], r['complexity_score']), reverse=True)
        rows = rows[:12]
        if not rows:
            continue
        cell_w, cell_h = 360, 280
        sheet = Image.new('RGB', (cell_w * 3, 70 + cell_h * 4), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 18), f'{tag}: representative experimental cases', fill=(20, 20, 20), font=title_font)
        for k, r in enumerate(rows):
            x = (k % 3) * cell_w
            y = 70 + (k // 3) * cell_h
            img = Image.open(OUT / r['copied_image']).convert('RGB')
            img.thumbnail((cell_w - 24, 185))
            sheet.paste(img, (x + 12, y + 8))
            draw.text((x + 12, y + 200), label(r), fill=(20, 20, 20), font=small)
        out = FIG_OUT / fname
        sheet.save(out)
        made.append(str(out.relative_to(OUT)))

    exemplars = []
    for tag in ['hierarchical_span_header', 'horizontal_dependency', 'vertical_dependency', 'long_range_dependency']:
        rows = [r for r in curated if tag in r['tags'] and r.get('copied_image')]
        rows.sort(key=lambda r: (r['complexity_score'], r['delta_full_vs_baseline']), reverse=True)
        if rows:
            exemplars.append((tag, rows[0]))
    for tag, r in exemplars:
        img = Image.open(OUT / r['copied_image']).convert('RGB')
        max_w = 1200
        if img.width > max_w:
            new_h = int(img.height * max_w / img.width)
            img = img.resize((max_w, new_h))
        caption_h = 140
        canvas = Image.new('RGB', (img.width, img.height + caption_h), 'white')
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        text = (
            f"{tag} | sample #{r['sample_index']} | {r['filename']}\n"
            f"baseline TEDS={r['baseline_teds']:.4f}, Dual-Branch/Phase5E TEDS={r['phase5e_teds']:.4f}, delta={r['delta_full_vs_baseline']:+.4f}\n"
            f"rows={r['row_count']}, max_cols={r['max_width']}, span_cells={r['span_tokens']}, rowspan={r['rowspan_tokens']}, colspan={r['colspan_tokens']}"
        )
        draw.text((18, img.height + 16), text, fill=(20, 20, 20), font=font)
        out = FIG_OUT / f"exemplar_{tag}_{str(r['sample_index']).zfill(5)}.png"
        canvas.save(out)
        made.append(str(out.relative_to(OUT)))
    (FIG_OUT / 'figure_manifest.json').write_text(json.dumps(made, indent=2, ensure_ascii=False), encoding='utf-8')
    return made

def main():
    records = build_records()
    curated = choose_curated(records, target=500)
    copied = copy_images(curated)
    write_jsonl(OUT / 'complex_structure_full_5000_metadata.jsonl', records)
    write_jsonl(OUT / 'complex_structure_curated_500_metadata.jsonl', curated)
    write_jsonl(OUT / 'complex_structure_curated_500_metadata_with_images.jsonl', curated)
    write_csv(curated)
    summary, counter, branch_counter, cur_counter = summarize(records, curated, copied)
    write_readme(summary, counter, branch_counter)
    made_figures = generate_figures(curated)
    summary['generated_figures'] = made_figures
    (OUT / 'dataset_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(OUT)

if __name__ == '__main__':
    main()
