import json, re, sys
from pathlib import Path

SKIP={"<html>","</html>","<body>","</body>","<thead>","</thead>","<tbody>","</tbody>"}
def merge(ts):
    out=[]; i=0
    while i<len(ts):
        t=str(ts[i])
        if t.startswith('<td') or t.startswith('<th'):
            v=t; i+=1
            while i<len(ts) and '>' not in v: v+=str(ts[i]); i+=1
            out.append(v)
        else: out.append(t); i+=1
    return out
def cells(ts, spans=True, occupancy=True, step_span=True):
    s=set(); occ=set(); r=-1; c=0; inside=False
    for t in merge([x for x in ts if str(x) not in SKIP]):
        if t.startswith('<tr'): r+=1; c=0; inside=True; continue
        if t.startswith('</tr'): inside=False; continue
        if not inside or not (t.startswith('<td') or t.startswith('<th')): continue
        if occupancy:
          while (r,c) in occ: c+=1
        rs=int(re.search(r'rowspan=["\']([0-9]+)',t).group(1)) if spans and re.search(r'rowspan=["\']([0-9]+)',t) else 1
        cs=int(re.search(r'colspan=["\']([0-9]+)',t).group(1)) if spans and re.search(r'colspan=["\']([0-9]+)',t) else 1
        s.add((r,c,rs,cs) if spans else (r,c))
        if occupancy:
          for rr in range(r,r+rs):
            for cc in range(c,c+cs): occ.add((rr,cc))
        c+=cs if step_span else 1
    return s
def run(p):
  micro=[]; macro=[]
  for line in open(p):
    x=json.loads(line); gt=x['gt_tokens']; pr=x['pred_tokens']
    for sp,oc,step in ((True,True,True),(False,True,True),(True,False,True),(False,False,True),(True,True,False),(True,False,False),(False,True,False),(False,False,False)):
      a=cells(pr,sp,oc,step); b=cells(gt,sp,oc,step); tp=len(a&b); fp=len(a-b); fn=len(b-a)
      micro.append(((sp,oc,step),tp,fp,fn))
      macro.append(((sp,oc,step),2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1))
  for key in ((True,True,True),(False,True,True),(True,False,True),(False,False,True),(True,True,False),(True,False,False),(False,True,False),(False,False,False)):
    q=[x for x in micro if x[0]==key]; tp=sum(x[1] for x in q); fp=sum(x[2] for x in q); fn=sum(x[3] for x in q)
    m=[x[1] for x in macro if x[0]==key]
    print(key,'micro',tp,fp,fn,2*tp/(2*tp+fp+fn),'macro',sum(m)/len(m))
for p in sys.argv[1:]: print('FILE',p); run(p)
