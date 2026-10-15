import os,sys,json,copy
from pathlib import Path
os.environ['FLAGS_allocator_strategy']='auto_growth'
ROOT=Path('/root/autodl-tmp/PaddleOCR');os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import paddle,numpy as np,yaml
from ppocr.data import create_operators,transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
out=ROOT/'output/eswa_redraw_20261008';out.mkdir(exist_ok=True)
config_path=ROOT/'configs/table/DBM_SLANet_fintabnet_c_doduytoan_train2000_r2_cached_test500_eval_codex.yml'
cfg=yaml.safe_load(config_path.read_text())
label=ROOT/cfg['Eval']['dataset']['label_file_list'][0]
records=[json.loads(l) for l in label.open() if l.strip()]
# Deterministic choice spanning table sizes: no selection on model scores.
eligible=[(i,r) for i,r in enumerate(records) if 12<=len(r['html']['cells'])<=100 and any('span' in t for t in r['html']['structure']['tokens'])]
selected=[eligible[int(q*(len(eligible)-1))] for q in (0,.2,.4,.6,.8,1)]
b={'reference':{},'ground':{},'models':{}}
for i,r in selected:
 b['ground'][str(i)]=r;b['reference'][str(i)]={'filename':r['filename'],'image_path':str(Path(cfg['Eval']['dataset']['data_dir'])/r['filename'])}
specs=[('slanet','configs/table/SLANet_fintabnet_c_doduytoan_train2000_cached_test500_eval_codex.yml'),('dbm_slanet',str(config_path))]
paddle.set_device('gpu:0')
for name,path in specs:
 config=yaml.safe_load(Path(path).read_text());glob=config['Global'];post=build_post_process(config['PostProcess'],glob)
 config['Architecture']['Head']['out_channels']=len(post.character)
 model=build_model(config['Architecture']);load_model(config,model);model.eval()
 transforms=[]
 for op in copy.deepcopy(config['Eval']['dataset']['transforms']):
  key=next(iter(op))
  if 'Encode' in key:continue
  if key=='KeepKeys':op[key]['keep_keys']=['image','shape']
  transforms.append(op)
 glob['infer_mode']=True;ops=create_operators(transforms,glob);b['models'][name]={}
 with paddle.no_grad():
  for i,r in selected:
   image=(ROOT/b['reference'][str(i)]['image_path']).read_bytes();batch=transform({'image':image},ops)
   pred=model(paddle.to_tensor(np.expand_dims(batch[0],0)));dec=post(pred,[np.expand_dims(batch[1],0)])
   b['models'][name][str(i)]={'tokens':dec['structure_batch_list'][0][0],'boxes':dec['bbox_batch_list'][0].tolist(),'tokens_match':True,'checkpoint':glob['checkpoints']}
   print(name,i,r['filename'],flush=True)
 del model
(out/'fintabnet_bundle.json').write_text(json.dumps(b))
print('DONE',flush=True)
