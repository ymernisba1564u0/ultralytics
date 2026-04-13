import argparse
from copy import deepcopy

from ultralytics import RTDETRDEIM
from ultralytics.engine.exporter import Exporter


def parse_args():
    p = argparse.ArgumentParser(description="Export RTDETRDEIM with upstream deploy conversion.")
    p.add_argument("weights", type=str, help="Path to RTDETRDEIM .pt checkpoint.")
    p.add_argument("--format", type=str, default="onnx", help="Export format.")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset.")
    p.add_argument("--device", default=0, help="CUDA device id or 'cpu'.")
    p.add_argument("--batch", type=int, default=1, help="Batch size.")
    p.add_argument("--half", action="store_true", help="FP16 export.")
    p.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph.")
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    # Deploy conversion is destructive (trims DFINE decoder layers, swaps weighting_function);
    # operate on a copy so the wrapper's live model is preserved.
    deploy_model = deepcopy(RTDETRDEIM(args.weights).model).eval().float()
    for p in deploy_model.parameters():
        p.requires_grad = False
    for m in deploy_model.modules():
        if hasattr(m, "convert_to_deploy"):
            m.convert_to_deploy()

    exporter = Exporter(overrides={
        "format": args.format,
        "imgsz": args.imgsz,
        "opset": args.opset,
        "device": args.device,
        "batch": args.batch,
        "half": args.half,
        "simplify": args.simplify,
    })
    print(exporter(model=deploy_model))


if __name__ == "__main__":
    main()
