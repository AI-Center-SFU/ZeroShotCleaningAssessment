from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image as load_lightglue_image
from lightglue.utils import rbd


ImageInput = Image.Image | str | Path


DEFAULT_DETECTION_CLASSES = [
    "a plastic bottle",
    "a glass bottle",
    "a glass jar",
    "a metal can",
    "a food can",
    "a drink can",
    "a food wrapper",
    "a candy wrapper",
    "a snack wrapper",
    "a foil wrapper",
    "a plastic packaging",
    "a plastic packet",
    "a snack bag",
    "a plastic shopping bag",
    "a paper bag",
    "a plastic container",
    "a food container",
    "a takeaway container",
    "a foam container",
    "a cardboard box",
    "a carton",
    "a milk carton",
    "a juice carton",
    "a food tray",
    "a plastic tray",
    "a disposable plate",
    "a paper plate",
    "a plastic plate",
    "a disposable bowl",
    "a cup",
    "a paper cup",
    "a plastic cup",
    "a lid",
    "a plastic lid",
    "a bottle cap",
    "a straw",
    "a cigarette",
    "a napkin",
    "a piece of paper",
]

DEFAULT_DETECTION_ALIASES = {
    "aluminum can": "metal can",
    "aluminium can": "metal can",
    "beverage can": "drink can",
    "can": "metal can",
    "wrapper": "food wrapper",
    "packet": "plastic packet",
    "package": "plastic packaging",
    "packaging": "plastic packaging",
    "plastic bag": "plastic shopping bag",
    "shopping bag": "plastic shopping bag",
    "plate": "disposable plate",
    "bowl": "disposable bowl",
    "tray": "food tray",
    "container": "plastic container",
    "box": "cardboard box",
    "paper": "piece of paper",
    "cap": "bottle cap",
}

BLOCKED_DETECTION_LABEL_TERMS = (
    "trash bag",
    "garbage bag",
    "waste bag",
    "rubbish bag",
    "black bag",
    "black plastic bag",
)


@dataclass
class ZeroShotCleaningAssessmentConfig:
    model_match: str = "lightglue"
    model_detect: str = "ensemble"
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    owl_vit_model: str = "google/owlvit-base-patch32"
    yolo_world_model: str = "yolov8x-world.pt"
    detection_classes: list[str] = field(default_factory=lambda: DEFAULT_DETECTION_CLASSES.copy())
    match_threshold: int = 77
    clean_threshold: float = 50.0
    box_threshold: float = 0.27
    text_threshold: float = 0.25
    ensemble_nms_threshold: float = 0.5
    max_keypoints: int = 2048
    device: str | None = None

    @property
    def detection_prompt(self) -> str:
        return ". ".join(self.detection_classes) + "."

    @property
    def allowed_detection_labels(self) -> set[str]:
        labels = set()
        for class_name in self.detection_classes:
            normalized = normalize_detection_label(class_name)
            labels.add(normalized)
            labels.add(normalized.removeprefix("a ").strip())
        labels.update(DEFAULT_DETECTION_ALIASES)
        return labels


@dataclass
class CleaningAssessmentResult:
    message: str
    matches_count: int = 0
    objects_before: int = 0
    objects_after: int = 0
    cleaning_score: float | None = None
    matched_points: list[dict[str, list[float]]] = field(default_factory=list)
    detections_before: list[dict[str, Any]] = field(default_factory=list)
    detections_after: list[dict[str, Any]] = field(default_factory=list)


class ZeroShotCleaningAssessment:
    def __init__(self, config: ZeroShotCleaningAssessmentConfig | None = None) -> None:
        self.config = config or ZeroShotCleaningAssessmentConfig()
        self.device = self._select_device()
        self.feature_extractor = None
        self.model_match = None
        self.processor = None
        self.model_detect = None
        self.grounding_dino_processor = None
        self.grounding_dino_model = None
        self.owl_vit_processor = None
        self.owl_vit_model = None
        self.yolo_world_model = None
        self.matched_points: list[dict[str, list[float]]] = []

    def assess(self, image_before: ImageInput, image_after: ImageInput) -> CleaningAssessmentResult:
        image_before = self.load_image(image_before)
        image_after = self.load_image(image_after)

        matches_count = self.match_images(image_before, image_after)
        if matches_count < self.config.match_threshold:
            return CleaningAssessmentResult(
                message="Изображения не сопоставимы",
                matches_count=matches_count,
                matched_points=self.matched_points,
            )

        detections_before = self.detect_objects(image_before)
        detections_after = self.detect_objects(image_after)

        objects_before = len(detections_before)
        objects_after = len(detections_after)

        if objects_before == 0:
            return CleaningAssessmentResult(
                message="Недостаточно данных для оценки",
                matches_count=matches_count,
                objects_before=objects_before,
                objects_after=objects_after,
                matched_points=self.matched_points,
                detections_before=detections_before,
                detections_after=detections_after,
            )

        cleaning_score = self.compute_cleaning_score(objects_before, objects_after)

        if cleaning_score >= self.config.clean_threshold:
            message = "Очистка признана эффективной"
        else:
            message = "Очистка признана недостаточной"

        return CleaningAssessmentResult(
            message=message,
            matches_count=matches_count,
            objects_before=objects_before,
            objects_after=objects_after,
            cleaning_score=cleaning_score,
            matched_points=self.matched_points,
            detections_before=detections_before,
            detections_after=detections_after,
        )

    def load_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def load_models(self) -> None:
        """Load image matching and zero-shot detection models."""
        self._load_match_model()
        self._load_detect_model()

    def _load_match_model(self) -> None:
        if self.model_match is None:
            self.feature_extractor = SuperPoint(
                max_num_keypoints=self.config.max_keypoints,
            ).eval().to(self.device)
            self.model_match = LightGlue(features="superpoint").eval().to(self.device)

    def _load_detect_model(self) -> None:
        if self.config.model_detect == "ensemble":
            self._load_ensemble_detect_models()
            return

        if self.model_detect is None:
            self.processor = AutoProcessor.from_pretrained(self.config.model_detect)
            self.model_detect = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.model_detect,
            ).eval().to(self.device)

    def _load_ensemble_detect_models(self) -> None:
        if self.grounding_dino_model is None:
            self.grounding_dino_processor = AutoProcessor.from_pretrained(
                self.config.grounding_dino_model,
            )
            self.grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.grounding_dino_model,
            ).eval().to(self.device)

        if self.owl_vit_model is None:
            self.owl_vit_processor = AutoProcessor.from_pretrained(self.config.owl_vit_model)
            self.owl_vit_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.owl_vit_model,
            ).eval().to(self.device)

        if self.yolo_world_model is None:
            from ultralytics import YOLOWorld

            self.yolo_world_model = YOLOWorld(self.config.yolo_world_model)
            self.yolo_world_model.set_classes(self.config.detection_classes)

    def match_images(self, image_before: Image.Image, image_after: Image.Image) -> int:
        """Return the number of matched keypoints between before/after images."""
        self._load_match_model()

        with TemporaryDirectory() as temp_dir:
            before_path = Path(temp_dir) / "before.jpg"
            after_path = Path(temp_dir) / "after.jpg"
            image_before.save(before_path)
            image_after.save(after_path)

            tensor_before = load_lightglue_image(before_path).to(self.device)
            tensor_after = load_lightglue_image(after_path).to(self.device)

            with torch.inference_mode():
                features_before = self.feature_extractor.extract(tensor_before)
                features_after = self.feature_extractor.extract(tensor_after)
                matches = self.model_match(
                    {"image0": features_before, "image1": features_after},
                )

            features_before, features_after, matches = [
                rbd(value) for value in (features_before, features_after, matches)
            ]
            keypoints_before = features_before["keypoints"]
            keypoints_after = features_after["keypoints"]
            match_indexes = matches["matches"]

            points_before = keypoints_before[match_indexes[..., 0]].detach().cpu().tolist()
            points_after = keypoints_after[match_indexes[..., 1]].detach().cpu().tolist()
            self.matched_points = [
                {"before": [float(x0), float(y0)], "after": [float(x1), float(y1)]}
                for (x0, y0), (x1, y1) in zip(points_before, points_after, strict=False)
            ]

            return len(self.matched_points)

    def detect_objects(self, image: Image.Image) -> list[dict[str, Any]]:
        """Return detected pollution objects for configured text classes."""
        self._load_detect_model()

        if self.config.model_detect == "ensemble":
            detections = []
            detections.extend(self._detect_grounding_dino(image))
            detections.extend(self._detect_owl_vit(image))
            detections.extend(self._detect_yolo_world(image))
            return nms_detections(detections, self.config.ensemble_nms_threshold)

        return self._detect_grounding_dino_single_model(image)

    def _detect_grounding_dino_single_model(self, image: Image.Image) -> list[dict[str, Any]]:
        inputs = self.processor(
            images=image,
            text=self.config.detection_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model_detect(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []
        labels = results.get("text_labels", results["labels"])
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            labels,
            strict=False,
        ):
            label = normalize_detection_label(label)
            if is_blocked_detection_label(label):
                continue
            if label not in self.config.allowed_detection_labels:
                continue
            label = DEFAULT_DETECTION_ALIASES.get(label, label)
            detections.append(
                {
                    "label": label,
                    "score": float(score.detach().cpu()),
                    "box": [float(value) for value in box.detach().cpu().tolist()],
                },
            )

        return detections

    def _detect_grounding_dino(self, image: Image.Image) -> list[dict[str, Any]]:
        inputs = self.grounding_dino_processor(
            images=image,
            text=self.config.detection_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.grounding_dino_model(**inputs)

        results = self.grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        return self._detections_from_grounded_results(results, source="grounding-dino")

    def _detect_owl_vit(self, image: Image.Image) -> list[dict[str, Any]]:
        inputs = self.owl_vit_processor(
            text=[self.config.detection_classes],
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.owl_vit_model(**inputs)

        results = self.owl_vit_processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.config.box_threshold,
            target_sizes=torch.tensor([image.size[::-1]], device=self.device),
            text_labels=[self.config.detection_classes],
        )[0]

        detections = []
        labels = results.get("text_labels", results["labels"])
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            labels,
            strict=False,
        ):
            if isinstance(label, torch.Tensor):
                label = self.config.detection_classes[int(label.detach().cpu())]
            detection = self._build_detection(box, score, label, source="owl-vit")
            if detection is not None:
                detections.append(detection)
        return detections

    def _detect_yolo_world(self, image: Image.Image) -> list[dict[str, Any]]:
        results = self.yolo_world_model.predict(
            image,
            conf=self.config.box_threshold,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        detections = []
        result = results[0]
        for box, score, class_id in zip(
            result.boxes.xyxy,
            result.boxes.conf,
            result.boxes.cls,
            strict=False,
        ):
            label = result.names[int(class_id.detach().cpu())]
            detection = self._build_detection(box, score, label, source="yolo-world")
            if detection is not None:
                detections.append(detection)
        return detections

    def _detections_from_grounded_results(
        self,
        results: dict[str, Any],
        source: str,
    ) -> list[dict[str, Any]]:
        detections = []
        labels = results.get("text_labels", results["labels"])
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            labels,
            strict=False,
        ):
            detection = self._build_detection(box, score, label, source=source)
            if detection is not None:
                detections.append(detection)
        return detections

    def _build_detection(
        self,
        box: torch.Tensor,
        score: torch.Tensor,
        label: object,
        source: str,
    ) -> dict[str, Any] | None:
        label = normalize_detection_label(label)
        if is_blocked_detection_label(label):
            return None
        if label not in self.config.allowed_detection_labels:
            return None
        label = DEFAULT_DETECTION_ALIASES.get(label, label)
        return {
            "label": label,
            "source": source,
            "score": float(score.detach().cpu()),
            "box": [float(value) for value in box.detach().cpu().tolist()],
        }

    def compute_cleaning_score(self, objects_before: int, objects_after: int) -> float:
        return (1 - objects_after / objects_before) * 100

    def _select_device(self) -> str:
        if self.config.device is not None:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def normalize_detection_label(label: object) -> str:
    return str(label).strip().lower()


def is_blocked_detection_label(label: str) -> bool:
    return any(term in label for term in BLOCKED_DETECTION_LABEL_TERMS)


def nms_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    detections = sorted(detections, key=lambda detection: detection["score"], reverse=True)
    kept = []

    while detections:
        current = detections.pop(0)
        kept.append(current)
        detections = [
            detection
            for detection in detections
            if box_iou(current["box"], detection["box"]) < iou_threshold
        ]

    return kept


def box_iou(box_a: list[float], box_b: list[float]) -> float:
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
    union_area = area_a + area_b - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area
