import json, importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("teds","eval_table_teds_codex.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
out=Path("output/phase6_teds_diagnostics_20260624")

def convert(tokens, mode):
    res=[]; i=0; hits=0
    while i<len(tokens):
        if tokens[i] == "<td":
            j=i+1; attrs=[]
            while j<len(tokens) and tokens[j] != ">":
                attrs.append(str(tokens[j])); j+=1
            if j<len(tokens) and tokens[j] == ">" and j+1<len(tokens) and tokens[j+1] == "</td>":
                attr=" ".join(attrs)
                has_row="rowspan" in attr; has_col="colspan" in attr
                if mode=="all" or (mode=="rowspan" and has_row) or (mode=="colspan" and has_col):
                    res.append("<td></td>"); i=j+2; hits+=1; continue
        res.append(tokens[i]); i+=1
    return res,hits

for file in ["phase5e_sample_teds.jsonl","phase5f_sample_teds.jsonl"]:
    rows=[json.loads(x) for x in (out/file).read_text().splitlines()]
    print("FILE",file)
    for mode in ["identity","all","rowspan","colspan"]:
        vals=[]; changed=improved=worsened=hits=0; details=[]
        for r in rows:
            pred=r["pred_tokens"]
            new,h=(pred,0) if mode=="identity" else convert(pred,mode)
            score=mod.teds_score(new,r["gt_tokens"])
            vals.append(score); hits+=h
            delta=score-r["teds"]
            if abs(delta)>1e-12:
                changed+=1; improved+=delta>0; worsened+=delta<0
                details.append((delta,r["sample_index"],r["filename"],r["teds"],score,h))
        print("MODE",mode,"TEDS",sum(vals)/len(vals),"hits",hits,"changed",changed,"improved",improved,"worsened",worsened)
        if mode!="identity":
            print("  worst",sorted(details)[:5])
            print("  best",sorted(details,reverse=True)[:5])
