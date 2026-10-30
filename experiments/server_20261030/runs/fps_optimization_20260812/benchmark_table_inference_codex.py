import json
import os
import statistics
import sys
import time

import numpy as np
import paddle


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("PADDLEOCR_ROOT", "/root/autodl-tmp/PaddleOCR")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
import tools.program as program


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype="float64"), q))


def summarize(values):
    return {
        "mean_ms": 1000.0 * statistics.fmean(values),
        "p50_ms": 1000.0 * percentile(values, 50),
        "p95_ms": 1000.0 * percentile(values, 95),
        "std_ms": 1000.0 * statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    config, device, logger, _ = program.preprocess()
    global_config = config["Global"]
    set_signal_handlers()

    dataloader = build_dataloader(config, "Eval", device, logger)
    post_process = build_post_process(config["PostProcess"], global_config)
    if hasattr(post_process, "character"):
        config["Architecture"]["Head"]["out_channels"] = len(post_process.character)

    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config["Architecture"]["model_type"])
    model.eval()

    warmup_batches = int(os.environ.get("FPS_BENCH_WARMUP", "20"))
    max_batches = int(os.environ.get("FPS_BENCH_BATCHES", "200"))
    output_path = os.environ["FPS_BENCH_OUTPUT"]
    run_id = os.environ.get("FPS_BENCH_RUN", "1")

    allocated_after_load = int(paddle.device.cuda.memory_allocated())
    reserved_after_load = int(paddle.device.cuda.memory_reserved())
    forward_times = []
    pipeline_times = []
    batch_sizes = []

    with paddle.no_grad():
        for idx, batch in enumerate(dataloader):
            images = batch[0]
            if idx < warmup_batches:
                _ = model(images, data=batch[1:])
                paddle.device.cuda.synchronize()
                continue
            if len(forward_times) >= max_batches:
                break

            paddle.device.cuda.synchronize()
            pipeline_start = time.perf_counter()
            forward_start = pipeline_start
            preds = model(images, data=batch[1:])
            paddle.device.cuda.synchronize()
            forward_end = time.perf_counter()

            batch_numpy = [
                item.numpy() if isinstance(item, paddle.Tensor) else item for item in batch
            ]
            _ = post_process(preds, batch_numpy)
            pipeline_end = time.perf_counter()

            forward_times.append(forward_end - forward_start)
            pipeline_times.append(pipeline_end - pipeline_start)
            batch_sizes.append(int(images.shape[0]))

    samples = sum(batch_sizes)
    forward_total = sum(forward_times)
    pipeline_total = sum(pipeline_times)
    result = {
        "run_id": run_id,
        "argv": sys.argv,
        "config": os.path.abspath(sys.argv[sys.argv.index("-c") + 1]),
        "checkpoint": global_config.get("checkpoints"),
        "head_config": config["Architecture"]["Head"],
        "warmup_batches": warmup_batches,
        "measured_batches": len(forward_times),
        "samples": samples,
        "batch_size_min": min(batch_sizes),
        "batch_size_max": max(batch_sizes),
        "forward_fps": samples / forward_total,
        "pipeline_fps": samples / pipeline_total,
        "forward_batch_latency": summarize(forward_times),
        "pipeline_batch_latency": summarize(pipeline_times),
        "allocated_after_load_bytes": allocated_after_load,
        "reserved_after_load_bytes": reserved_after_load,
        "peak_allocated_bytes": int(paddle.device.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(paddle.device.cuda.max_memory_reserved()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
