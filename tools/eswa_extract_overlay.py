import json
import os
import sys
from pathlib import Path

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
ROOT = Path('/root/autodl-tmp/PaddleOCR')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
import paddle
import yaml
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model

OUT = ROOT / 'output/eswa_overlay_20261008'
OUT.mkdir(exist_ok=True)
INDICES = [48, 1158, 2436, 4052, 4313]
models = ['original_horizontal', 'original_no_bidir']
reference = {}
with open(ROOT / 'output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['sample_index'] in INDICES:
            reference[r['sample_index']] = r
names = {r['filename']: i for i, r in reference.items()}
ground = {}
with open(ROOT / 'train_data/table/pubtabnet/phase10_val_5000.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['filename'] in names:
            ground[names[r['filename']]] = r
bundle = json.loads((OUT / 'bundle.json').read_text())
paddle.set_device('gpu:0')
for name in models:
    source = 'true_horizontal' if name == 'original_horizontal' else 'full_mamba_one_way'
    config = yaml.safe_load((ROOT / f'output/main_tables_rebuild_20260810/configs/{source}.yml').read_text())
    if name == 'original_horizontal':
        config = yaml.safe_load((ROOT / 'configs/table/DBM_SLANet_expC_horizontal_mamba_10000_from_phase5e_codex.yml').read_text())
        config['Global']['pretrained_model'] = None
        config['Global']['checkpoints'] = './output/E10C_horizontal_mamba_10000_from_phase5e_20260624/best_accuracy'
        diagnostic = 'output/phase10_axis_ablation_val5000_20260710/expC_horizontal_val5000_sample_teds.jsonl'
    else:
        config['Global']['checkpoints'] = './output/E11E_full_mamba_lambda02_bidirfalse_30000_from_baseline_20260709/best_accuracy'
        diagnostic = 'output/phase10_more_models_val5000_20260710/e11e_bidir_false_sample_teds.jsonl'
    glob = config['Global']
    post = build_post_process(config['PostProcess'], glob)
    config['Architecture']['Head']['out_channels'] = len(post.character)
    model = build_model(config['Architecture'])
    load_model(config, model)
    model.eval()
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        key = next(iter(op))
        if 'Encode' in key:
            continue
        if key == 'KeepKeys':
            op[key]['keep_keys'] = ['image', 'shape']
        transforms.append(op)
    glob['infer_mode'] = True
    ops = create_operators(transforms, glob)
    stored = {}
    with open(ROOT / diagnostic) as f:
        for line in f:
            r = json.loads(line)
            if r['sample_index'] in INDICES:
                stored[r['sample_index']] = r
    results = {}
    with paddle.no_grad():
        for idx, rec in reference.items():
            image = (ROOT / rec['image_path']).read_bytes()
            batch = transform({'image': image}, ops)
            pred = model(paddle.to_tensor(np.expand_dims(batch[0], 0)))
            decoded = post(pred, [np.expand_dims(batch[1], 0)])
            tokens = decoded['structure_batch_list'][0][0]
            results[idx] = {'tokens': tokens, 'boxes': decoded['bbox_batch_list'][0].tolist(),
                            'stored_tokens': stored[idx]['pred_tokens'],
                            'stored_teds': stored[idx]['teds'],
                            'tokens_match': tokens == stored[idx]['pred_tokens']}
    bundle['models'][name] = results
    (OUT / 'bundle.json').write_text(json.dumps(bundle))
    print(name, [(i, r['tokens_match']) for i, r in results.items()], flush=True)
    del model
print('DONE', flush=True)
