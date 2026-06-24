
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

base = Path("output/phase7_span_diagnostics_20260624")
phase5e_path = Path("output/phase6_teds_diagnostics_20260624/phase5e_sample_teds.jsonl")
phase7b_path = base / "phase7b_sample_teds.jsonl"
phase5e = [json.loads(line) for line in phase5e_path.read_text(encoding="utf-8").splitlines() if line.strip()]
phase7b = [json.loads(line) for line in phase7b_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def span_counts(tokens):
    return {
        "rowspan": sum("rowspan" in str(t) for t in tokens),
        "colspan": sum("colspan" in str(t) for t in tokens),
        "span_total": sum(("rowspan" in str(t)) or ("colspan" in str(t)) for t in tokens),
    }

def token_counts(tokens):
    c = Counter(tokens)
    td_open = sum(str(t).startswith("<td") for t in tokens)
    s = span_counts(tokens)
    return {
        "len": len(tokens),
        "tr": c["<tr>"],
        "td_open": td_open,
        "td_close": c["</td>"],
        "empty_td": c["<td></td>"],
        "rowspan": s["rowspan"],
        "colspan": s["colspan"],
        "span_total": s["span_total"],
    }

def compact(tokens, max_chars=260):
    text = " ".join(map(str, tokens))
    return text if len(text) <= max_chars else text[:max_chars] + " ..."

def first_diff(left, right):
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            return tag, i1, i2, j1, j2, left[max(0, i1-8):min(len(left), i2+8)], right[max(0, j1-8):min(len(right), j2+8)]
    return None

rows = []
for e, b in zip(phase5e, phase7b):
    gt = e["gt_tokens"]
    e_pred = e["pred_tokens"]
    b_pred = b["pred_tokens"]
    e_span = span_counts(e_pred)
    b_span = span_counts(b_pred)
    gt_span = span_counts(gt)
    rows.append({
        "sample_index": e["sample_index"],
        "filename": e.get("filename", ""),
        "image_path": e.get("image_path", ""),
        "phase5e_teds": e["teds"],
        "phase7b_teds": b["teds"],
        "delta_7b_minus_5e": b["teds"] - e["teds"],
        "phase5e_exact": e["exact"],
        "phase7b_exact": b["exact"],
        "phase5e_len": e["pred_len"],
        "phase7b_len": b["pred_len"],
        "gt_len": e["gt_len"],
        "gt_span_total": gt_span["span_total"],
        "phase5e_span_total": e_span["span_total"],
        "phase7b_span_total": b_span["span_total"],
        "phase5e_extra_spans": max(0, e_span["span_total"] - gt_span["span_total"]),
        "phase7b_extra_spans": max(0, b_span["span_total"] - gt_span["span_total"]),
        "phase5e_missing_spans": max(0, gt_span["span_total"] - e_span["span_total"]),
        "phase7b_missing_spans": max(0, gt_span["span_total"] - b_span["span_total"]),
    })

base.mkdir(parents=True, exist_ok=True)
(base / "phase5e_vs_phase7b_teds_delta.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in sorted(rows, key=lambda x: x["delta_7b_minus_5e"])) + "\n",
    encoding="utf-8",
)

worse = [r for r in rows if r["delta_7b_minus_5e"] < -1e-12]
better = [r for r in rows if r["delta_7b_minus_5e"] > 1e-12]
same = len(rows) - len(worse) - len(better)
mean_delta = sum(r["delta_7b_minus_5e"] for r in rows) / len(rows)
mean_worse = sum(r["delta_7b_minus_5e"] for r in worse) / max(1, len(worse))
mean_better = sum(r["delta_7b_minus_5e"] for r in better) / max(1, len(better))
span_summary = {
    "phase5e_extra_span_samples": sum(r["phase5e_extra_spans"] > 0 for r in rows),
    "phase7b_extra_span_samples": sum(r["phase7b_extra_spans"] > 0 for r in rows),
    "phase5e_missing_span_samples": sum(r["phase5e_missing_spans"] > 0 for r in rows),
    "phase7b_missing_span_samples": sum(r["phase7b_missing_spans"] > 0 for r in rows),
    "phase5e_total_extra_spans": sum(r["phase5e_extra_spans"] for r in rows),
    "phase7b_total_extra_spans": sum(r["phase7b_extra_spans"] for r in rows),
    "phase5e_total_missing_spans": sum(r["phase5e_missing_spans"] for r in rows),
    "phase7b_total_missing_spans": sum(r["phase7b_missing_spans"] for r in rows),
}

def table_rows(items):
    return ["| {sample_index} | {delta_7b_minus_5e:.6f} | {phase5e_teds:.6f} | {phase7b_teds:.6f} | {phase5e_exact} | {phase7b_exact} | {phase5e_len}/{phase7b_len}/{gt_len} | {phase5e_span_total}/{phase7b_span_total}/{gt_span_total} | {filename} |".format(**r) for r in items]

md = [
    "# Phase7-B Span-Aware Loss Diagnostic Analysis",
    "",
    "Comparison: Phase5-E current best vs Phase7-B weak span-aware loss on the same 1000-sample validation split.",
    "",
    "## Summary",
    "",
    f"- Samples: {len(rows)}",
    f"- Same TEDS: {same}",
    f"- Phase7-B worse: {len(worse)}, mean delta {mean_worse:.6f}",
    f"- Phase7-B better: {len(better)}, mean delta {mean_better:.6f}",
    f"- Overall mean delta: {mean_delta:.12f}",
    f"- Exact matches: Phase5-E {sum(r['phase5e_exact'] for r in rows)}, Phase7-B {sum(r['phase7b_exact'] for r in rows)}",
    "",
    "## Span Count Summary",
    "",
    "| Metric | Phase5-E | Phase7-B |",
    "|---|---:|---:|",
    f"| Samples with extra predicted spans | {span_summary['phase5e_extra_span_samples']} | {span_summary['phase7b_extra_span_samples']} |",
    f"| Total extra predicted spans | {span_summary['phase5e_total_extra_spans']} | {span_summary['phase7b_total_extra_spans']} |",
    f"| Samples with missing spans | {span_summary['phase5e_missing_span_samples']} | {span_summary['phase7b_missing_span_samples']} |",
    f"| Total missing spans | {span_summary['phase5e_total_missing_spans']} | {span_summary['phase7b_total_missing_spans']} |",
    "",
    "## Worst Phase7-B Regressions",
    "",
    "| sample | delta 7B-5E | 5E TEDS | 7B TEDS | 5E exact | 7B exact | len 5E/7B/GT | spans 5E/7B/GT | filename |",
    "|---:|---:|---:|---:|---|---|---:|---:|---|",
]
md.extend(table_rows(sorted(rows, key=lambda x: x["delta_7b_minus_5e"])[:15]))
md.extend(["", "## Best Phase7-B Improvements", "", "| sample | delta 7B-5E | 5E TEDS | 7B TEDS | 5E exact | 7B exact | len 5E/7B/GT | spans 5E/7B/GT | filename |", "|---:|---:|---:|---:|---|---|---:|---:|---|"])
md.extend(table_rows(sorted(rows, key=lambda x: x["delta_7b_minus_5e"], reverse=True)[:10]))
md.extend(["", "## Token-Level Examples", ""])
for title, items in [("Worst regressions", sorted(rows, key=lambda x: x["delta_7b_minus_5e"])[:5]), ("Best improvements", sorted(rows, key=lambda x: x["delta_7b_minus_5e"], reverse=True)[:5])]:
    md.append(f"### {title}")
    md.append("")
    for r in items:
        idx = r["sample_index"]
        e = phase5e[idx]
        b = phase7b[idx]
        gt = e["gt_tokens"]
        e_pred = e["pred_tokens"]
        b_pred = b["pred_tokens"]
        md.append(f"#### sample {idx}: {r['filename']}")
        md.append("")
        md.append(f"- delta 7B-5E: {r['delta_7b_minus_5e']:.6f}")
        md.append(f"- TEDS 5E/7B: {r['phase5e_teds']:.6f} / {r['phase7b_teds']:.6f}")
        md.append(f"- counts 5E: `{token_counts(e_pred)}`")
        md.append(f"- counts 7B: `{token_counts(b_pred)}`")
        md.append(f"- counts GT: `{token_counts(gt)}`")
        diff = first_diff(e_pred, b_pred)
        if diff:
            tag, i1, i2, j1, j2, lc, rc = diff
            md.append(f"- first 5E-vs-7B diff: `{tag}` 5E[{i1}:{i2}] vs 7B[{j1}:{j2}]")
            md.append(f"  - 5E context: `{compact(lc)}`")
            md.append(f"  - 7B context: `{compact(rc)}`")
        diff_gt = first_diff(b_pred, gt)
        if diff_gt:
            tag, i1, i2, j1, j2, lc, rc = diff_gt
            md.append(f"- first 7B-vs-GT diff: `{tag}` 7B[{i1}:{i2}] vs GT[{j1}:{j2}]")
            md.append(f"  - 7B context: `{compact(lc)}`")
            md.append(f"  - GT context: `{compact(rc)}`")
        md.append("")
md.extend(["## Interpretation", "", "Phase7-B confirms that simple span-token reweighting is not an effective improvement path in the current setup.", "", "- It increases exact-match accuracy from 0.693 to 0.697, but lowers Structure-TEDS from 0.830817 to 0.828003.", "- The result repeats the Phase5-F pattern: exact-match accuracy can improve while the tree-edit objective gets worse.", "- The remaining bottleneck is not just detecting span tokens, but deciding when spans are structurally safe. Future work should move toward span confidence calibration or constrained decoding rather than stronger loss reweighting."])
(base / "phase5e_vs_phase7b_span_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("WROTE", base / "phase5e_vs_phase7b_span_analysis.md")
print("WROTE", base / "phase5e_vs_phase7b_teds_delta.jsonl")
print("same", same, "worse", len(worse), "better", len(better), "mean_delta", mean_delta)
print(span_summary)
