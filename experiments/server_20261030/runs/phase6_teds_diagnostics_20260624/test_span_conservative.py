import json
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("teds", "eval_table_teds_codex.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

out = Path("output/phase6_teds_diagnostics_20260624")
rows = [json.loads(x) for x in (out/"phase5f_sample_teds.jsonl").read_text(encoding="utf-8").splitlines()]


def is_span_td(tok):
    s = str(tok)
    return s.startswith("<td") and ("rowspan" in s or "colspan" in s)


def td_empty_to_plain(tokens, mode):
    res = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if is_span_td(tok):
            # Structure labels encode an empty span cell as: <td ... > </td>.
            if i + 1 < len(tokens) and tokens[i + 1] == "</td>":
                if mode == "all_empty_spans":
                    res.append("<td></td>")
                    i += 2
                    continue
                if mode == "rowspan_only" and "rowspan" in str(tok):
                    res.append("<td></td>")
                    i += 2
                    continue
                if mode == "colspan_only" and "colspan" in str(tok):
                    res.append("<td></td>")
                    i += 2
                    continue
        res.append(tok)
        i += 1
    return res

modes = ["identity", "all_empty_spans", "rowspan_only", "colspan_only"]
summary = []
for mode in modes:
    vals = []
    changed = 0
    improved = 0
    worsened = 0
    details = []
    for r in rows:
        pred = r["pred_tokens"]
        gt = r["gt_tokens"]
        new_pred = pred if mode == "identity" else td_empty_to_plain(pred, mode)
        score = mod.teds_score(new_pred, gt)
        vals.append(score)
        delta = score - r["teds"]
        if abs(delta) > 1e-12:
            changed += 1
            improved += int(delta > 0)
            worsened += int(delta < 0)
            details.append((delta, r["sample_index"], r["filename"], r["teds"], score, len(pred), len(new_pred)))
    summary.append((mode, sum(vals)/len(vals), changed, improved, worsened, sorted(details)[:5], sorted(details, reverse=True)[:5]))

for mode, mean, changed, improved, worsened, worst, best in summary:
    print("MODE", mode, "TEDS", mean, "changed", changed, "improved", improved, "worsened", worsened)
    if mode != "identity":
        print("  worst", worst)
        print("  best", best)
