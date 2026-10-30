import json
from html.parser import HTMLParser
from pathlib import Path
import sys
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import to_rgba
from PIL import Image

sys.path.insert(0, r'C:\Users\cczcz\.codex\skills\nature-figure\scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).parent
OUT = Path(r'D:\ESWA\figures\eswa_cell_overlays')
OUT.mkdir(exist_ok=True)
B = json.loads((ROOT / 'bundle.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'hatch.linewidth': .22})
COLORS = ['#67B9B6', '#DC74A0', '#E3B85E']
LABELS = ['Body cell', 'Header cell', 'Spanning body cell']

class Cells(HTMLParser):
    def __init__(self):
        super().__init__(); self.header = False; self.kinds = []
    def handle_starttag(self, tag, attrs):
        if tag == 'thead': self.header = True
        if tag in ('td', 'th'):
            a = dict(attrs)
            span = int(a.get('rowspan', 1)) > 1 or int(a.get('colspan', 1)) > 1
            self.kinds.append(1 if self.header or tag == 'th' else (2 if span else 0))
    def handle_endtag(self, tag):
        if tag == 'thead': self.header = False

def kinds(tokens):
    p = Cells(); p.feed(''.join(tokens)); return p.kinds

def gt(idx):
    record = B['ground'][str(idx)]
    cats = kinds(record['html']['structure']['tokens'])
    assert len(cats) == len(record['html']['cells'])
    return [(cell['bbox'], k) for cell, k in zip(record['html']['cells'], cats) if cell.get('bbox')]

def prediction(idx, model):
    record = B['models'][model][str(idx)]
    assert record['tokens_match'], (idx, model)
    cats = kinds(record['tokens'])
    assert len(cats) == len(record['boxes']), (idx, model, len(cats),len(record['boxes']))
    return list(zip(record['boxes'], cats))

def draw(ax, idx, model, title, fontsize):
    im = Image.open(ROOT / B['reference'][str(idx)]['filename']).convert('RGB')
    ax.imshow(im, interpolation='bilinear')
    cells = gt(idx) if model is None else prediction(idx, model)
    invalid = 0
    for box, category in cells:
        x1,y1,x2,y2 = box
        if x2 <= x1 or y2 <= y1:
            invalid += 1; continue
        color = COLORS[category]
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,
                              facecolor=to_rgba(color,.14), edgecolor=to_rgba(color,.85),
                              linewidth=.45, hatch='////', clip_on=True,rasterized=True))
    ax.set_xlim(-.5,im.width-.5);ax.set_ylim(im.height-.5,-.5)
    ax.set_aspect('equal');ax.axis('off')
    ax.set_title(title,fontsize=fontsize,pad=5,color='#202020')
    return invalid

def legend(ax, size):
    ax.axis('off')
    handles = [Patch(facecolor=to_rgba(c,.14),edgecolor=c,hatch='////',label=t,rasterized=True)
               for c,t in zip(COLORS,LABELS)]
    ax.legend(handles=handles,loc='center',ncol=3,fontsize=size,
              handlelength=1.6,columnspacing=1.1,handletextpad=.4,
              frameon=True,edgecolor='#dddddd',fancybox=False,borderpad=.35)
    for patch in ax.get_legend().get_patches(): patch.set_rasterized(True)

def export(fig, stem, axes, groups=None):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(OUT / (stem+'.alignment.json')),
                                      tolerance_pt=1.5,strict=True,axes=axes,
                                      column_groups=[['a','b']])
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/(stem+'.'+ext),dpi=350,facecolor='white')
    plt.close(fig)

def pair(idx, model, title, stem):
    im = Image.open(ROOT / B['reference'][str(idx)]['filename'])
    w = 7.2; h = w*.94*im.height/im.width
    fig = plt.figure(figsize=(w,2*h+1.55))
    axes = []
    for row,(m,t) in enumerate([(None,'(a) Ground truth'),(model,'(b) '+title)]):
        bottom = .6 + (1-row)*(h+.6)
        ax=fig.add_axes([.03,bottom/fig.get_figheight(),.94,h/fig.get_figheight()])
        draw(ax,idx,m,t,11);axes.append(ax)
        lg=fig.add_axes([.02,(bottom-.28)/fig.get_figheight(),.96,.24/fig.get_figheight()]);legend(lg,8)
    export(fig,stem,axes)

MODELS = [('slanet','SLANet'),('cell_token','Cell Token'),('lite_mamba','Lite Mamba'),
          ('full_mamba_no_presence','Mamba (no presence)'),('full_mamba_presence','Mamba (presence)'),
          ('full_mamba_one_way','Mamba (one-way)'),('true_horizontal','Horizontal-only'),
          ('true_vertical','Vertical-only'),('zero_geometry','Mamba (zero geometry)'),
          ('dbm_d64','DBM (d=64)'),('dbm_ds8','DBM (state=8)'),('dbm_d192','DBM (d=192)'),
          ('dbm_e1','DBM (e=1)'),('dbm_slanet','DBM-SLANet (ours)')]
FIVE=[MODELS[0],MODELS[1],MODELS[2],('original_horizontal','Horizontal Mamba'),MODELS[-1]]
ABL=[('lite_mamba','Lite Mamba'),('original_no_bidir','No bidirectional'),('full_mamba_presence','Full model')]
for model,title in MODELS: pair(2436,model,title,'fig5_'+model)
for idx,letter in [(4313,'a'),(48,'b')]:
    for model,title in FIVE: pair(idx,model,title,'fig9_'+letter+'_'+model)
for idx,letter,variants in [(48,'c',ABL),(4052,'d',[ABL[0],('zero_geometry','No geometry'),ABL[-1]])]:
    for model,title in variants:pair(idx,model,title,'fig10_'+letter+'_'+model)

# Separate model panels retain the exact input aspect ratio within each case.
def sheet(cases,stem,cols):
    nrows=sum((len(v)+1+cols-1)//cols for _,_,v in cases)
    cellw=3.0
    ratios=[Image.open(ROOT/B['reference'][str(idx)]['filename']).height/
            Image.open(ROOT/B['reference'][str(idx)]['filename']).width for idx,_,_ in cases]
    rowh=cellw*max(ratios)+.65
    fig=plt.figure(figsize=(cols*cellw,nrows*rowh))
    rowoff=0
    image_axes=[]
    row_groups=[]
    for (idx,label,variants),ratio in zip(cases,ratios):
        panels=[(None,'Ground truth')]+variants
        group=[]
        for j,(model,title) in enumerate(panels):
            row=rowoff+j//cols;col=j%cols
            h=cellw*.94*ratio
            bottom=fig.get_figheight()-(row+1)*rowh+.3
            ax=fig.add_axes([(col*cellw+.09)/fig.get_figwidth(),bottom/fig.get_figheight(),
                             cellw*.94/fig.get_figwidth(),h/fig.get_figheight()])
            draw(ax,idx,model,(label+' - ' if label else '')+title,9)
            image_axes.append(ax)
            group.append('p'+str(len(image_axes)))
            if len(group)==cols or j==len(panels)-1:
                if len(group)>1:row_groups.append(group)
                group=[]
            lg=fig.add_axes([(col*cellw)/fig.get_figwidth(),(bottom-.24)/fig.get_figheight(),
                            cellw/fig.get_figwidth(),.2/fig.get_figheight()]);legend(lg,5.4)
        rowoff+=(len(panels)+cols-1)//cols
    # Different case ratios are intentional; measure within-case plot rectangles.
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=image_axes,
          panel_ids=['p'+str(i+1) for i in range(len(image_axes))],row_groups=row_groups,
          json_out=str(OUT/(stem+'.alignment.json')),strict=True)
    rectangles=[{'bounds':list(ax.get_position().bounds)} for ax in fig.axes if len(ax.images)]
    (OUT/(stem+'.layout.json')).write_text(json.dumps(rectangles,indent=2))
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(stem+'.'+ext),dpi=350,facecolor='white')
    plt.close(fig)

sheet([(2436,'',MODELS)],'fig5',4)
sheet([(4313,'(a)',FIVE),(48,'(b)',FIVE)],'fig9_ab_parallel',3)
sheet([(48,'(c)',ABL),(4052,'(d)',[ABL[0],('zero_geometry','No geometry'),ABL[-1]])],
      'fig10_cd_parallel',2)
(OUT/'README.txt').write_text('Source: PubTabNet content bounding boxes and checkpoint localization outputs.\n'
    'Empty GT cells without a box are not drawn; no invented full-cell boundaries.\n'
    'Colors: body / HTML thead header / spanning body cell. No projected-row-header labels exist.\n'
    'No weights, training data, evaluation tables, or manuscript text changed.\n',encoding='utf-8')
print('Rendered 30 standalone pairs and three comparison sheets:',OUT)
