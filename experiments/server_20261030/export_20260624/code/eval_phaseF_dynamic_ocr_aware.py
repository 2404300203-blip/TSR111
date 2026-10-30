import argparse
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import paddle

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppstructure.table.convert_label2html import gen_html
from ppstructure.table.matcher import TableMatch
from ppstructure.table.table_metric.table_metric import TEDS
from tools.infer.predict_det import TextDetector
from tools.infer.predict_rec import TextRecognizer
from tools.infer.predict_system import sorted_boxes
import tools.program as program


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    h, w = shape[:2]
    return max(x0 - pix, 0), max(y0 - pix, 0), min(x1 + pix, w), min(y1 + pix, h)


class DynamicTableStructurer:
    def __init__(self, config):
        self.config = config
        self.global_config = config["Global"]
        self.post = build_post_process(config["PostProcess"], self.global_config)
        if hasattr(self.post, "character"):
            config["Architecture"]["Head"]["out_channels"] = len(self.post.character)
        self.model = build_model(config["Architecture"])
        load_model(config, self.model, model_type=config["Architecture"]["model_type"])
        self.model.eval()
        self.preprocess_op = create_operators([
            {"ResizeTableImage": {"max_len": 488}},
            {"NormalizeImage": {"scale": "1./255.", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "order": "hwc"}},
            {"PaddingTableImage": {"size": [488, 488]}},
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ])

    def __call__(self, img):
        tic = time.time()
        data = transform({"image": img}, self.preprocess_op)
        if data is None:
            return None, 0.0
        image = np.expand_dims(data[0], axis=0)
        shape = np.expand_dims(data[-1], axis=0)
        with paddle.no_grad():
            preds = self.model(paddle.to_tensor(image))
        post_result = self.post(preds, [shape])
        structure = post_result["structure_batch_list"][0][0]
        bboxes = post_result["bbox_batch_list"][0]
        structure = ["<html>", "<body>", "<table>"] + structure + ["</table>", "</body>", "</html>"]
        return (structure, bboxes), time.time() - tic


class OCRRunner:
    def __init__(self, args):
        self.detector = TextDetector(copy.deepcopy(args))
        self.recognizer = TextRecognizer(copy.deepcopy(args))

    def __call__(self, img):
        h, w = img.shape[:2]
        dt_boxes, det_t = self.detector(copy.deepcopy(img))
        if dt_boxes is None:
            return np.array([]), [], det_t, 0.0
        dt_boxes = sorted_boxes(dt_boxes)
        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            r_boxes.append([x_min, y_min, x_max, y_max])
        dt_boxes = np.array(r_boxes)
        crops = []
        for box in dt_boxes:
            x0, y0, x1, y1 = expand(2, box, img.shape)
            crops.append(img[int(y0):int(y1), int(x0):int(x1), :])
        rec_res, rec_t = self.recognizer(crops) if crops else ([], 0.0)
        return dt_boxes, rec_res, det_t, rec_t


def make_ocr_args(args):
    return SimpleNamespace(
        use_gpu=True, use_xpu=False, use_npu=False, use_mlu=False, use_gcu=False, use_metax_gpu=False,
        ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision="fp32", gpu_mem=500, gpu_id=0,
        image_dir=None, page_num=0,
        det_algorithm="DB", det_model_dir=args.det_model_dir, det_limit_side_len=960, det_limit_type="max", det_box_type="quad",
        det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode="fast",
        det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2,
        det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1,
        scales=[8,16,32], alpha=1.0, beta=1.0, fourier_degree=5,
        rec_algorithm="SVTR_LCNet", rec_model_dir=args.rec_model_dir, rec_image_inverse=True, rec_image_shape="3, 48, 320", rec_batch_num=6,
        max_text_length=25, rec_char_dict_path="./ppocr/utils/en_dict.txt", use_space_char=True, vis_font_path="./doc/fonts/simfang.ttf", drop_score=0.5,
        use_angle_cls=False, cls_model_dir=None, cls_image_shape="3, 48, 192", label_list=["0","180"], cls_batch_num=6, cls_thresh=0.9,
        enable_mkldnn=None, cpu_threads=10, use_pdserving=False, warmup=False,
        sr_model_dir=None, sr_image_shape="3, 32, 128", sr_batch_num=1,
        draw_img_save_dir="./inference_results", save_crop_res=False, crop_res_save_dir="./output",
        use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path="./log_output/", show_log=False,
        use_onnx=False, onnx_providers=False, onnx_sess_options=False, return_word_box=False,
    )


def load_records(label_file, data_dir, limit):
    rows = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            info = json.loads(line)
            rows.append((info["filename"], os.path.join(data_dir, info["filename"]), gen_html(info)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--det_model_dir", required=True)
    ap.add_argument("--rec_model_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    config = program.load_config(args.config)
    config = program.merge_config(config, {"Global.checkpoints": args.checkpoint, "Global.pretrained_model": None})
    paddle.set_device("gpu:0")
    os.makedirs(args.output, exist_ok=True)

    eval_cfg = config["Eval"]["dataset"]
    records = load_records(eval_cfg["label_file_list"][0], eval_cfg["data_dir"], args.limit)
    structurer = DynamicTableStructurer(config)
    ocr = OCRRunner(make_ocr_args(args))
    matchers = {
        "legacy": TableMatch(filter_ocr_result=True, match_mode="legacy"),
        "ocr_aware_hungarian": TableMatch(
            filter_ocr_result=True,
            match_mode="ocr_aware_hungarian",
            diagnostics_path=os.path.join(args.output, "matcher_diagnostics.jsonl"),
        ),
    }
    pred_htmls = {k: [] for k in matchers}
    gt_htmls = []
    times = {k: 0.0 for k in ["table", "det", "rec", "legacy_match", "ocr_aware_hungarian_match"]}
    sample_path = os.path.join(args.output, "samples.jsonl")
    with open(sample_path, "w", encoding="utf-8") as fw:
        for idx, (name, img_path, gt_html) in enumerate(records):
            img = cv2.imread(img_path)
            if img is None:
                continue
            structure_res, table_t = structurer(img)
            dt_boxes, rec_res, det_t, rec_t = ocr(img)
            times["table"] += table_t
            times["det"] += det_t
            times["rec"] += rec_t
            gt_htmls.append(gt_html)
            row = {"index": idx, "filename": name, "ocr_boxes": int(len(dt_boxes)), "rec_tokens": len(rec_res)}
            for mode, matcher in matchers.items():
                tic = time.time()
                pred = matcher(structure_res, dt_boxes, rec_res)
                times[f"{mode}_match"] += time.time() - tic
                pred_htmls[mode].append(pred)
                row[f"{mode}_empty_td"] = pred.count("<td></td>")
                row[f"{mode}_html"] = pred
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{idx+1}/{len(records)}] {name} ocr={len(dt_boxes)}", flush=True)

    teds = TEDS(n_jobs=8)
    summary = {"samples": len(gt_htmls), "sample_path": sample_path, "times": times}
    for mode in matchers:
        scores = teds.batch_evaluate_html(gt_htmls, pred_htmls[mode])
        summary[mode] = {
            "teds": sum(scores) / max(len(scores), 1),
            "empty_td_total": sum(x.count("<td></td>") for x in pred_htmls[mode]),
        }
    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
