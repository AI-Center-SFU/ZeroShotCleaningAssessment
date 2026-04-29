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
    model_detect: str = "IDEA-Research/grounding-dino-base"
    detection_classes: list[str] = field(default_factory=lambda: DEFAULT_DETECTION_CLASSES.copy())
    match_threshold: int = 77
    clean_threshold: float = 50.0
    box_threshold: float = 0.27
    text_threshold: float = 0.25
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
        if self.model_detect is None:
            self.processor = AutoProcessor.from_pretrained(self.config.model_detect)
            self.model_detect = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.model_detect,
            ).eval().to(self.device)

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
