import sys; sys.path.insert(0,"/root/autodl-tmp/PaddleOCR")
import csv, json
from company_cell_audit import parse_cells
from pathlib import Path
pairs=[]
for model,base in [('SLANet','output/company_cell_f1_20260914/predictions'),('DBM-SLANet','output/deployment_export_20260812/financial/predictions')]:
  for s in ['20260806','20260807','20260808']:
    p=Path(base)/f'company_seed{s}_{"slanet" if model=="SLANet" else "full"}.jsonl'
    rows=[]
    with p.open() as f:
      for line in f:
        x=json.loads(line); pred=parse_cells(x.get('pred_tokens',[])); gt=parse_cells(x.get('gt_tokens',[])); tp=len(pred&gt); fp=len(pred-gt); fn=len(gt-pred); den=2*tp+fp+fn
        rows.append([model,s,x.get('sample_index'),x.get('filename',''),tp,fp,fn,2*tp/den if den else 1.0,len(pred),len(gt)])
    pairs.extend(rows)
out='output/company_cell_f1_20260914/company_cell_f1_per_sample.csv'
with open(out,'w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['model','seed','sample_index','filename','tp','fp','fn','cell_f1','pred_cells','gt_cells']); w.writerows(pairs)
print(out, len(pairs))
