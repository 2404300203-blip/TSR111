import json
from pathlib import Path

out = Path("output/phase6_teds_diagnostics_20260624")
e = [json.loads(x) for x in (out/"phase5e_sample_teds.jsonl").read_text(encoding="utf-8").splitlines()]
f = [json.loads(x) for x in (out/"phase5f_sample_teds.jsonl").read_text(encoding="utf-8").splitlines()]

def counts(tokens):
    return {
        "len": len(tokens),
        "tr": tokens.count("<tr>"),
        "td_open": sum(1 for t in tokens if str(t).startswith("<td")),
        "td_close": tokens.count("</td>"),
        "rowspan": sum(1 for t in tokens if "rowspan" in str(t)),
        "colspan": sum(1 for t in tokens if "colspan" in str(t)),
        "empty": tokens.count("<td></td>"),
    }

def mean(rows): return sum(rows)/len(rows)
base_e = mean([r["teds"] for r in e])
base_f = mean([r["teds"] for r in f])
oracle = mean([max(a["teds"], b["teds"]) for a,b in zip(e,f)])
print("base_e", base_e)
print("base_f", base_f)
print("oracle_best_of_EF", oracle, "gain_vs_E", oracle-base_e)

changed=[]
for a,b in zip(e,f):
    if abs(a["teds"]-b["teds"])>1e-12:
        ca, cb, cg = counts(a["pred_tokens"]), counts(b["pred_tokens"]), counts(a["gt_tokens"])
        changed.append((a,b,ca,cb,cg))
print("changed", len(changed))
print("idx delta e f conf_e conf_f cE cF cGT file")
for a,b,ca,cb,cg in sorted(changed, key=lambda x: x[1]["teds"]-x[0]["teds"]):
    print(a["sample_index"], round(b["teds"]-a["teds"],6), round(a["teds"],6), round(b["teds"],6), round(a.get("confidence") or 0,6), round(b.get("confidence") or 0,6), ca, cb, cg, a["filename"])

# Test a few inference-available selectors. No GT used.
selectors = {}
selectors["higher_conf"] = [b if (b.get("confidence") or 0) > (a.get("confidence") or 0) else a for a,b in zip(e,f)]
selectors["prefer_f_if_more_span"] = [b if (counts(b["pred_tokens"])["rowspan"]+counts(b["pred_tokens"])["colspan"]) > (counts(a["pred_tokens"])["rowspan"]+counts(a["pred_tokens"])["colspan"]) else a for a,b in zip(e,f)]
selectors["prefer_f_if_shorter"] = [b if b["pred_len"] < a["pred_len"] else a for a,b in zip(e,f)]
selectors["prefer_f_if_exact_conf_margin_001"] = [b if (b.get("confidence") or 0) > (a.get("confidence") or 0)+0.01 else a for a,b in zip(e,f)]
for name, chosen in selectors.items():
    val = mean([r["teds"] for r in chosen])
    print("selector", name, val, "gain_vs_E", val-base_e, "chosen_F", sum(1 for r,b in zip(chosen,f) if r is b))
