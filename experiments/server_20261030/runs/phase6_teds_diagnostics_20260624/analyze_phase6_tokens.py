import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

out = Path("output/phase6_teds_diagnostics_20260624")
e_rows = [
    json.loads(line)
    for line in (out / "phase5e_sample_teds.jsonl").read_text(encoding="utf-8").splitlines()
]
f_rows = [
    json.loads(line)
    for line in (out / "phase5f_sample_teds.jsonl").read_text(encoding="utf-8").splitlines()
]
delta_rows = [
    json.loads(line)
    for line in (out / "phase5e_vs_phase5f_teds_delta.jsonl").read_text(encoding="utf-8").splitlines()
]
worst = [row for row in delta_rows if row["delta_f_minus_e"] < -1e-12][:7]
best = sorted(
    [row for row in delta_rows if row["delta_f_minus_e"] > 1e-12],
    key=lambda row: row["delta_f_minus_e"],
    reverse=True,
)


def tag_counts(tokens):
    counts = Counter(tokens)
    td_open = sum(1 for token in tokens if str(token).startswith("<td"))
    return {
        "len": len(tokens),
        "tr": counts["<tr>"],
        "td_open": td_open,
        "td_close": counts["</td>"],
        "empty_td_token": counts["<td></td>"],
        "rowspan": sum(1 for token in tokens if "rowspan" in str(token)),
        "colspan": sum(1 for token in tokens if "colspan" in str(token)),
        "thead": counts["<thead>"],
        "tbody": counts["<tbody>"],
    }


def compact(tokens, max_chars=240):
    text = " ".join(map(str, tokens))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def first_diff(left, right):
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            left_context = left[max(0, i1 - 8) : min(len(left), i2 + 8)]
            right_context = right[max(0, j1 - 8) : min(len(right), j2 + 8)]
            return tag, i1, i2, j1, j2, left_context, right_context
    return None


def add_section(lines, title, rows):
    lines.append(f"## {title}")
    lines.append("")
    for row in rows:
        idx = row["sample_index"]
        e_row = e_rows[idx]
        f_row = f_rows[idx]
        gt_tokens = e_row["gt_tokens"]
        e_tokens = e_row["pred_tokens"]
        f_tokens = f_row["pred_tokens"]
        lines.append(f"### sample {idx}: {row['filename']}")
        lines.append("")
        lines.append(f"- delta F-E: {row['delta_f_minus_e']:.6f}")
        lines.append(f"- E/F TEDS: {row['e_teds']:.6f} / {row['f_teds']:.6f}")
        lines.append(f"- E exact / F exact: {row['e_exact']} / {row['f_exact']}")
        lines.append(f"- E counts: `{tag_counts(e_tokens)}`")
        lines.append(f"- F counts: `{tag_counts(f_tokens)}`")
        lines.append(f"- GT counts: `{tag_counts(gt_tokens)}`")

        diff = first_diff(e_tokens, f_tokens)
        if diff:
            tag, i1, i2, j1, j2, left_context, right_context = diff
            lines.append(f"- first E-vs-F diff: `{tag}` E[{i1}:{i2}] vs F[{j1}:{j2}]")
            lines.append(f"  - E context: `{compact(left_context)}`")
            lines.append(f"  - F context: `{compact(right_context)}`")

        diff_gt = first_diff(f_tokens, gt_tokens)
        if diff_gt:
            tag, i1, i2, j1, j2, left_context, right_context = diff_gt
            lines.append(f"- first F-vs-GT diff: `{tag}` F[{i1}:{i2}] vs GT[{j1}:{j2}]")
            lines.append(f"  - F context: `{compact(left_context)}`")
            lines.append(f"  - GT context: `{compact(right_context)}`")
        lines.append("")


md = [
    "# Phase6 Token Difference Analysis",
    "",
    "This report compares Phase5-E current best and Phase5-F rejected predictions on samples whose TEDS changed.",
    "",
]
add_section(md, "Worst Phase5-F Regressions", worst)
add_section(md, "Phase5-F Improvements", best)

target = out / "phase5e_vs_phase5f_token_diff_analysis.md"
target.write_text("\n".join(md), encoding="utf-8")
print(f"WROTE {target}")
print("\n".join(md[:90]))
