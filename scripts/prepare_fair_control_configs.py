#!/usr/bin/env python3
"""Generate matched-control configs on the remote PaddleOCR checkout."""

from pathlib import Path
import copy
import yaml

ROOT = Path('/root/autodl-tmp/PaddleOCR')
BASE = ROOT / 'configs/table/DBM_SLANet_phase11c_full_mamba_lambda02_30000_from_baseline_codex.yml'
OUT = ROOT / 'output/fair_controls_20260806/configs'
TRAIN_LABEL = 'train_data/table/pubtabnet/phase4_stage5_train_30000.jsonl'
CTA_LABEL = 'train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl'

VARIANTS = {
    'cell_token': {'head': 'CellTokenMLPHead', 'loss': 'SLACellTokenLoss'},
    'mamba_dual_bidir': {'head': 'CellFullMambaHead', 'axis_mode': 'dual', 'bidirectional': True},
    'mamba_dual_unidir': {'head': 'CellFullMambaHead', 'axis_mode': 'dual', 'bidirectional': False},
    'mamba_horizontal_bidir': {'head': 'CellFullMambaHead', 'axis_mode': 'horizontal', 'bidirectional': True},
    'mamba_vertical_bidir': {'head': 'CellFullMambaHead', 'axis_mode': 'vertical', 'bidirectional': True},
    'slanet_loc10': {'head': 'SLAHead', 'loss': 'SLALoss'},
}


def patch_config(name, spec):
    with BASE.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    run = ROOT / f'output/fair_controls_20260806/{name}'
    cfg['Global'].update({
        'seed': 20260806,
        'epoch_num': 1,
        'save_model_dir': str(run),
        'save_inference_dir': str(run / 'infer'),
        'save_res_path': str(run / 'infer'),
        'pretrained_model': './output/baseline_package_20260621/model/best_accuracy.pdparams',
        'checkpoints': None,
    })
    cfg['Optimizer']['lr']['learning_rate'] = 2e-6
    cfg['Optimizer']['lr']['values'] = [2e-6, 1e-6, 5e-7]
    cfg['Train']['loader']['batch_size_per_card'] = 4
    cfg['Train']['loader']['num_workers'] = 0
    cfg['Train']['dataset']['label_file_list'] = [TRAIN_LABEL]
    cfg['Eval']['loader']['batch_size_per_card'] = 4
    cfg['Eval']['loader']['num_workers'] = 0
    cfg['Eval']['dataset']['label_file_list'] = [CTA_LABEL]
    head = cfg['Architecture']['Head']
    head['name'] = spec['head']
    head.pop('axis_mode', None)
    head.pop('use_gt_bbox_mask', None)
    loss = cfg['Loss']
    loss['name'] = spec.get('loss', 'SLACellTokenLoss')
    loss['loc_weight'] = 10.0
    loss['cell_presence_weight'] = 0.2 if loss['name'] == 'SLACellTokenLoss' else None
    if loss['cell_presence_weight'] is None:
        loss.pop('cell_presence_weight', None)
    if spec['head'] == 'CellFullMambaHead':
        head['axis_mode'] = spec['axis_mode']
        head['bidirectional'] = spec['bidirectional']
        head['use_gt_bbox_mask'] = False
    cfg['Global']['eval_batch_step'] = [0, 99999999]
    path = OUT / f'{name}.yml'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


if __name__ == '__main__':
    for name, spec in VARIANTS.items():
        print(patch_config(name, spec))
