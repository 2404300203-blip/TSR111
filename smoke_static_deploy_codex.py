import hashlib
import json
import os

import cv2
import numpy as np
from paddle import inference


ROOT = "/root/autodl-tmp/PaddleOCR"
MODEL_DIR = os.path.join(ROOT, "output/deployment_export_20260812/inference")
IMAGE = os.path.join(ROOT, "train_data/table/pubtabnet/val/PMC5755158_010_01.png")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    scale = min(488.0 / max(height, width), 1.0)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)))
    canvas = np.zeros((488, 488, 3), dtype="float32")
    canvas[: resized.shape[0], : resized.shape[1]] = resized.astype("float32")
    canvas /= 255.0
    canvas -= np.asarray([0.485, 0.456, 0.406], dtype="float32")
    canvas /= np.asarray([0.229, 0.224, 0.225], dtype="float32")
    return canvas.transpose(2, 0, 1)[None]


def main():
    model_path = os.path.join(MODEL_DIR, "inference.pdmodel")
    params_path = os.path.join(MODEL_DIR, "inference.pdiparams")
    config = inference.Config(model_path, params_path)
    config.enable_use_gpu(512, 0)
    config.switch_ir_optim(True)
    predictor = inference.create_predictor(config)
    inputs = predictor.get_input_names()
    image = preprocess(IMAGE)
    handle = predictor.get_input_handle(inputs[0])
    handle.reshape(image.shape)
    handle.copy_from_cpu(image)
    predictor.run()
    outputs = {}
    for name in predictor.get_output_names():
        value = predictor.get_output_handle(name).copy_to_cpu()
        outputs[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all()),
            "sum": float(value.astype("float64").sum()),
        }
    result = {
        "image": IMAGE,
        "model": model_path,
        "params": params_path,
        "model_sha256": sha256(model_path),
        "params_sha256": sha256(params_path),
        "input_names": inputs,
        "output_names": predictor.get_output_names(),
        "outputs": outputs,
        "success": all(item["finite"] for item in outputs.values()),
    }
    path = os.path.join(ROOT, "output/deployment_export_20260812/static_inference_smoke.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
