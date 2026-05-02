from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


PROMPT_CLASSES = [
    "a plastic bottle",
    "a glass bottle",
    "a metal can",
    "a plastic bag",
    "a food wrapper",
    "a paper bag",
    "a plastic container",
    "a cup",
    "a straw",
    "a cigarette",
]

PROMPT_TO_CATEGORY_IDS = {
    "a plastic bottle": {4, 5},
    "a glass bottle": {6},
    "a metal can": {10, 11, 12},
    "a plastic bag": {38, 40, 41},
    "a food wrapper": {36, 39, 42},
    "a paper bag": {34, 35},
    "a plastic container": {43, 44, 45, 46, 47},
    "a cup": {20, 21, 22, 23, 24},
    "a straw": {55, 56},
    "a cigarette": {59},
}

CATEGORY_ID_TO_PROMPT = {
    category_id: prompt_class
    for prompt_class, category_ids in PROMPT_TO_CATEGORY_IDS.items()
    for category_id in category_ids
}

LABEL_ALIASES = {
    "plastic bottle": "a plastic bottle",
    "glass bottle": "a glass bottle",
    "metal can": "a metal can",
    "can": "a metal can",
    "food can": "a metal can",
    "drink can": "a metal can",
    "a food can": "a metal can",
    "a drink can": "a metal can",
    "plastic bag": "a plastic bag",
    "bag": "a plastic bag",
    "food wrapper": "a food wrapper",
    "wrapper": "a food wrapper",
    "plastic wrapper": "a food wrapper",
    "paper bag": "a paper bag",
    "plastic container": "a plastic container",
    "container": "a plastic container",
    "cup": "a cup",
    "straw": "a straw",
    "cigarette": "a cigarette",
}

DEFAULT_MODELS = {
    "grounding-dino": "IDEA-Research/grounding-dino-base",
    "owl-vit": "google/owlvit-base-patch32",
    "yolo-world": "yolov8x-world.pt",
}


@dataclass(frozen=True)
class GroundTruthBox:
    image_id: int
    prompt_class: str
    box_xyxy: tuple[float, float, float, float]
    matched: bool = False


@dataclass(frozen=True)
class Prediction:
    image_id: int
    prompt_class: str
    raw_label: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class EvaluationConfig:
    annotations: Path
    image_root: Path
    output_dir: Path
    backend: str = "grounding-dino"
    model: str | None = None
    prompt_classes: list[str] | None = None
    box_threshold: float = 0.27
    text_threshold: float = 0.25
    iou_threshold: float = 0.5
    gt_scope: str = "all"
    class_aware: bool = False
    limit: int | None = None
    device: str | None = None
    save_predictions: bool = True
    use_cache: bool = True
    verbose: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot litter detection on the local COCO annotations.",
    )
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/zero_shot_detection"))
    parser.add_argument(
        "--backend",
        choices=("grounding-dino", "owl-vit", "yolo-world"),
        default="grounding-dino",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id/path. Defaults depend on --backend.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.27)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gt-scope",
        choices=("all", "prompt"),
        default="all",
        help="all evaluates against every COCO litter annotation; prompt uses only mapped classes.",
    )
    parser.add_argument(
        "--class-aware",
        action="store_true",
        help="Require predicted prompt class to match mapped GT class. Implies --gt-scope prompt.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        annotations=args.annotations,
        image_root=args.image_root,
        output_dir=args.output_dir,
        backend=args.backend,
        model=args.model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        iou_threshold=args.iou_threshold,
        gt_scope=args.gt_scope,
        class_aware=args.class_aware,
        limit=args.limit,
        device=args.device,
        save_predictions=args.save_predictions,
        use_cache=args.use_cache,
        verbose=True,
    )
    run_evaluation(config)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    if config.class_aware and config.gt_scope != "prompt":
        config = EvaluationConfig(**{**config.__dict__, "gt_scope": "prompt"})

    model_name = config.model or DEFAULT_MODELS[config.backend]
    prompt_classes = config.prompt_classes or PROMPT_CLASSES
    metrics_path = config.output_dir / "metrics.json"
    predictions_path = config.output_dir / "predictions.csv"

    if config.use_cache and metrics_path.exists():
        if config.verbose:
            print(f"Cached: {metrics_path}")
        return read_json(metrics_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(config.device)
    coco = read_json(config.annotations)
    images = sorted(coco["images"], key=lambda item: int(item["id"]))
    if config.limit is not None:
        images = images[: config.limit]

    image_ids = {int(image["id"]) for image in images}
    gt_by_image = build_ground_truth(coco["annotations"], image_ids, gt_scope=config.gt_scope)
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())

    if config.verbose:
        print(f"Backend: {config.backend}")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Images: {len(images)}")
        print(f"GT boxes: {total_gt}")
        print(f"Prompt: {detection_prompt(prompt_classes)}")

    detector = load_detector(config.backend, model_name, device, prompt_classes)

    predictions: list[Prediction] = []
    started_at = time.perf_counter()

    for index, image_info in enumerate(images, start=1):
        image_path = config.image_root / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image file is missing: {image_path}")

        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        predictions.extend(
            detect_image(
                detector=detector,
                backend=config.backend,
                image=image,
                image_id=int(image_info["id"]),
                prompt_classes=prompt_classes,
                box_threshold=config.box_threshold,
                text_threshold=config.text_threshold,
            ),
        )
        if config.verbose:
            print(f"{index}/{len(images)} {image_info['file_name']}", flush=True)

    elapsed_seconds = time.perf_counter() - started_at
    metrics = evaluate_predictions(
        predictions=predictions,
        gt_by_image=gt_by_image,
        iou_threshold=config.iou_threshold,
        class_aware=config.class_aware,
    )
    metrics.update(
        {
            "model": model_name,
            "backend": config.backend,
            "annotations": str(config.annotations),
            "images": len(images),
            "gt_boxes": total_gt,
            "predicted_boxes": len(predictions),
            "elapsed_seconds": elapsed_seconds,
            "images_per_second": safe_divide(len(images), elapsed_seconds),
            "box_threshold": config.box_threshold,
            "text_threshold": config.text_threshold,
            "iou_threshold": config.iou_threshold,
            "gt_scope": config.gt_scope,
            "class_aware": config.class_aware,
            "prompt": detection_prompt(prompt_classes),
            "prompt_classes": prompt_classes,
        },
    )

    write_json(metrics_path, metrics)
    if config.save_predictions:
        write_predictions(predictions_path, predictions)

    if config.verbose:
        print_summary(metrics, metrics_path)

    return metrics


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_ground_truth(
    annotations: list[dict[str, Any]],
    image_ids: set[int],
    gt_scope: str,
) -> dict[int, list[GroundTruthBox]]:
    gt_by_image: dict[int, list[GroundTruthBox]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_ids:
            continue
        if gt_scope == "prompt":
            if category_id not in CATEGORY_ID_TO_PROMPT:
                continue
            prompt_class = CATEGORY_ID_TO_PROMPT[category_id]
        else:
            prompt_class = "litter"
        gt_by_image[image_id].append(
            GroundTruthBox(
                image_id=image_id,
                prompt_class=prompt_class,
                box_xyxy=coco_bbox_to_xyxy(annotation["bbox"]),
            ),
        )
    return dict(gt_by_image)


def coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in bbox]
    return (x, y, x + width, y + height)


def load_detector(
    backend: str,
    model_name: str,
    device: str,
    prompt_classes: list[str],
) -> Any:
    if backend in {"grounding-dino", "owl-vit"}:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
        model.eval()
        return {"processor": processor, "model": model}

    if backend == "yolo-world":
        from ultralytics import YOLOWorld

        model = YOLOWorld(model_name)
        model.set_classes(prompt_classes)
        return {"model": model, "device": device}

    raise ValueError(f"Unsupported backend: {backend}")


def detect_image(
    detector: Any,
    backend: str,
    image: Image.Image,
    image_id: int,
    prompt_classes: list[str],
    box_threshold: float,
    text_threshold: float,
) -> list[Prediction]:
    if backend == "grounding-dino":
        return detect_grounding_dino(
            detector=detector,
            image=image,
            image_id=image_id,
            prompt_classes=prompt_classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    if backend == "owl-vit":
        return detect_owl_vit(
            detector=detector,
            image=image,
            image_id=image_id,
            prompt_classes=prompt_classes,
            box_threshold=box_threshold,
        )
    if backend == "yolo-world":
        return detect_yolo_world(
            detector=detector,
            image=image,
            image_id=image_id,
            box_threshold=box_threshold,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def detect_grounding_dino(
    detector: dict[str, Any],
    image: Image.Image,
    image_id: int,
    prompt_classes: list[str],
    box_threshold: float,
    text_threshold: float,
) -> list[Prediction]:
    model = detector["model"]
    processor = detector["processor"]
    device = str(model.device)
    inputs = processor(images=image, text=detection_prompt(prompt_classes), return_tensors="pt").to(
        device,
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    labels = result.get("text_labels", result["labels"])
    predictions = []
    for box, score, label in zip(result["boxes"], result["scores"], labels, strict=False):
        raw_label = str(label).strip()
        prompt_class = normalize_prediction_label(label)
        if prompt_class is None:
            continue
        predictions.append(
            Prediction(
                image_id=image_id,
                prompt_class=prompt_class,
                raw_label=raw_label,
                score=float(score.detach().cpu()),
                box_xyxy=tuple(float(value) for value in box.detach().cpu().tolist()),
            ),
        )
    return predictions


def detect_owl_vit(
    detector: dict[str, Any],
    image: Image.Image,
    image_id: int,
    prompt_classes: list[str],
    box_threshold: float,
) -> list[Prediction]:
    model = detector["model"]
    processor = detector["processor"]
    device = str(model.device)
    inputs = processor(text=[prompt_classes], images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=box_threshold,
        target_sizes=torch.tensor([image.size[::-1]], device=device),
        text_labels=[prompt_classes],
    )[0]

    predictions = []
    labels = result.get("text_labels", result.get("labels"))
    for box, score, label in zip(result["boxes"], result["scores"], labels, strict=False):
        if isinstance(label, torch.Tensor):
            prompt_class = prompt_classes[int(label.detach().cpu())]
        else:
            prompt_class = normalize_prediction_label(label)
            if prompt_class is None:
                continue
        predictions.append(
            Prediction(
                image_id=image_id,
                prompt_class=prompt_class,
                raw_label=str(label).strip(),
                score=float(score.detach().cpu()),
                box_xyxy=tuple(float(value) for value in box.detach().cpu().tolist()),
            ),
        )
    return predictions


def detect_yolo_world(
    detector: dict[str, Any],
    image: Image.Image,
    image_id: int,
    box_threshold: float,
) -> list[Prediction]:
    model = detector["model"]
    device = detector["device"]
    results = model.predict(image, conf=box_threshold, device=device, verbose=False)
    if not results:
        return []

    result = results[0]
    predictions = []
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return []

    for box, score, class_id in zip(
        boxes.xyxy,
        boxes.conf,
        boxes.cls,
        strict=False,
    ):
        raw_label = str(names[int(class_id.detach().cpu())])
        prompt_class = normalize_prediction_label(raw_label)
        if prompt_class is None:
            continue
        predictions.append(
            Prediction(
                image_id=image_id,
                prompt_class=prompt_class,
                raw_label=raw_label,
                score=float(score.detach().cpu()),
                box_xyxy=tuple(float(value) for value in box.detach().cpu().tolist()),
            ),
        )
    return predictions


def detection_prompt(prompt_classes: list[str] | None = None) -> str:
    return ". ".join(prompt_classes or PROMPT_CLASSES) + "."


def normalize_prediction_label(label: object) -> str | None:
    normalized = str(label).strip().lower()
    normalized = " ".join(normalized.replace(".", " ").split())
    if normalized in PROMPT_CLASSES:
        return normalized
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]
    without_article = normalized.removeprefix("a ").strip()
    if without_article in LABEL_ALIASES:
        return LABEL_ALIASES[without_article]
    return None


def evaluate_predictions(
    predictions: list[Prediction],
    gt_by_image: dict[int, list[GroundTruthBox]],
    iou_threshold: float,
    class_aware: bool,
) -> dict[str, float | int]:
    gt_matched: dict[int, set[int]] = defaultdict(set)
    tp = 0
    fp = 0

    predictions = sorted(predictions, key=lambda prediction: prediction.score, reverse=True)
    for prediction in predictions:
        candidates = gt_by_image.get(prediction.image_id, [])
        best_index = None
        best_iou = 0.0
        for index, gt_box in enumerate(candidates):
            if index in gt_matched[prediction.image_id]:
                continue
            if class_aware and gt_box.prompt_class != prediction.prompt_class:
                continue
            current_iou = iou(prediction.box_xyxy, gt_box.box_xyxy)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            tp += 1
            gt_matched[prediction.image_id].add(best_index)
        else:
            fp += 1

    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    fn = total_gt - tp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return safe_divide(intersection_area, area_a + area_b - intersection_area)


def select_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(path: Path, predictions: list[Prediction]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_id", "prompt_class", "raw_label", "score", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "image_id": prediction.image_id,
                    "prompt_class": prediction.prompt_class,
                    "raw_label": prediction.raw_label,
                    "score": prediction.score,
                    "x1": prediction.box_xyxy[0],
                    "y1": prediction.box_xyxy[1],
                    "x2": prediction.box_xyxy[2],
                    "y2": prediction.box_xyxy[3],
                },
            )


def print_summary(metrics: dict[str, Any], metrics_path: Path) -> None:
    print(f"Predicted boxes: {metrics['predicted_boxes']}")
    print(f"Images/sec: {metrics['images_per_second']:.4f}")
    print(f"GT scope: {metrics['gt_scope']} class-aware: {metrics['class_aware']}")
    print(
        f"Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}",
    )
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
