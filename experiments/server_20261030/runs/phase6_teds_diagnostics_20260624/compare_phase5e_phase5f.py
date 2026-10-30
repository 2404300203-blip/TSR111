import json
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

rows = []
for e, f in zip(e_rows, f_rows):
    rows.append(
        {
            "sample_index": e["sample_index"],
            "filename": e.get("filename", ""),
            "image_path": e.get("image_path", ""),
            "e_teds": e["teds"],
            "f_teds": f["teds"],
            "delta_f_minus_e": f["teds"] - e["teds"],
            "e_exact": e["exact"],
            "f_exact": f["exact"],
            "e_len": e["pred_len"],
            "f_len": f["pred_len"],
            "gt_len": e["gt_len"],
        }
    )

rows_sorted = sorted(rows, key=lambda row: row["delta_f_minus_e"])
(out / "phase5e_vs_phase5f_teds_delta.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows_sorted) + "\n",
    encoding="utf-8",
)

worse = [row for row in rows if row["delta_f_minus_e"] < -1e-12]
better = [row for row in rows if row["delta_f_minus_e"] > 1e-12]
same = len(rows) - len(worse) - len(better)
mean_worse = sum(row["delta_f_minus_e"] for row in worse) / max(len(worse), 1)
mean_better = sum(row["delta_f_minus_e"] for row in better) / max(len(better), 1)
mean_delta = sum(row["delta_f_minus_e"] for row in rows) / len(rows)

md = [
    "# Phase6 Phase5-E vs Phase5-F TEDS Delta Analysis",
    "",
    "## Summary",
    "",
    f"- Samples: {len(rows)}",
    f"- Same TEDS: {same}",
    f"- Phase5-F worse: {len(worse)}, mean delta {mean_worse:.6f}",
    f"- Phase5-F better: {len(better)}, mean delta {mean_better:.6f}",
    f"- Overall mean delta: {mean_delta:.12f}",
    f"- Exact matches: Phase5-E {sum(row['e_exact'] for row in rows)}, Phase5-F {sum(row['f_exact'] for row in rows)}",
    "",
    "## Worst Phase5-F Regressions",
    "",
    "| sample | delta F-E | E TEDS | F TEDS | E exact | F exact | len E/F/GT | filename |",
    "|---:|---:|---:|---:|---|---|---:|---|",
]
for row in rows_sorted[:15]:
    md.append(
        "| {sample_index} | {delta_f_minus_e:.6f} | {e_teds:.6f} | {f_teds:.6f} | "
        "{e_exact} | {f_exact} | {e_len}/{f_len}/{gt_len} | {filename} |".format(**row)
    )

md.extend(
    [
        "",
        "## Best Phase5-F Improvements",
        "",
        "| sample | delta F-E | E TEDS | F TEDS | E exact | F exact | len E/F/GT | filename |",
        "|---:|---:|---:|---:|---|---|---:|---|",
    ]
)
for row in sorted(rows, key=lambda item: item["delta_f_minus_e"], reverse=True)[:10]:
    md.append(
        "| {sample_index} | {delta_f_minus_e:.6f} | {e_teds:.6f} | {f_teds:.6f} | "
        "{e_exact} | {f_exact} | {e_len}/{f_len}/{gt_len} | {filename} |".format(**row)
    )

(out / "phase5e_vs_phase5f_delta_summary.md").write_text(
    "\n".join(md) + "\n", encoding="utf-8"
)
print("\n".join(md[:45]))
