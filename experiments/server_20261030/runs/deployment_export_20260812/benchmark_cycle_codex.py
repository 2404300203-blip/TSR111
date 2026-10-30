import json
import os
import statistics
import sys
import time

import numpy as np
import paddle


ROOT = "/root/autodl-tmp/PaddleOCR"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
import tools.program as program


def summary(values):
    values = np.asarray(values, dtype="float64")
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
        "std_ms": float(values.std(ddof=1) * 1000.0) if len(values) > 1 else 0.0,
    }


def main():
    config, device, logger, _ = program.preprocess()
    global_config = config["Global"]
    set_signal_handlers()
    loader = build_dataloader(config, "Eval", device, logger)
    post_process = build_post_process(config["PostProcess"], global_config)
    if hasattr(post_process, "character"):
        config["Architecture"]["Head"]["out_channels"] = len(post_process.character)
    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config["Architecture"]["model_type"])
    model.eval()

    warmup = int(os.environ.get("FPS_BENCH_WARMUP", "20"))
    measured = int(os.environ.get("FPS_BENCH_BATCHES", "100"))
    output = os.environ["FPS_BENCH_OUTPUT"]
    forward_times, pipeline_times, sizes = [], [], []
    iteration = 0
    with paddle.no_grad():
        while len(forward_times) < measured:
            for batch in loader:
                images = batch[0]
                if iteration < warmup:
                    _ = model(images, data=batch[1:])
                    paddle.device.cuda.synchronize()
                    iteration += 1
                    continue
                paddle.device.cuda.synchronize()
                start = time.perf_counter()
                preds = model(images, data=batch[1:])
                paddle.device.cuda.synchronize()
                forward_end = time.perf_counter()
                batch_numpy = [x.numpy() if isinstance(x, paddle.Tensor) else x for x in batch]
                _ = post_process(preds, batch_numpy)
                end = time.perf_counter()
                forward_times.append(forward_end - start)
                pipeline_times.append(end - start)
                sizes.append(int(images.shape[0]))
                iteration += 1
                if len(forward_times) >= measured:
                    break
    samples = sum(sizes)
    result = {
        "config": os.path.abspath(sys.argv[sys.argv.index("-c") + 1]),
        "checkpoint": global_config.get("checkpoints"),
        "run_id": os.environ.get("FPS_BENCH_RUN", "1"),
        "warmup_batches": warmup,
        "measured_batches": len(forward_times),
        "samples": samples,
        "batch_size_min": min(sizes),
        "batch_size_max": max(sizes),
        "forward_fps": samples / sum(forward_times),
        "pipeline_fps": samples / sum(pipeline_times),
        "forward_batch_latency": summary(forward_times),
        "pipeline_batch_latency": summary(pipeline_times),
        "peak_allocated_bytes": int(paddle.device.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(paddle.device.cuda.max_memory_reserved()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
