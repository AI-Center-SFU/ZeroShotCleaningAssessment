from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image as load_lightglue_image
from lightglue.utils import rbd


FILENAME_RE = re.compile(r"(?:IMG|PXL)_(\d{8})_(\d{9})")


@dataclass(frozen=True)
class Photo:
    path: Path
    timestamp: datetime
    has_annotation: bool
    group_id: int


@dataclass(frozen=True)
class PhotoPair:
    group_id: int
    before: Photo
    after: Photo
    negative: Photo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("poop-2020-12-28"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/poop_keypoints"))
    parser.add_argument("--group-gap-seconds", type=int, default=180)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    photos = load_photos(args.dataset, args.group_gap_seconds)
    pairs = build_pairs(photos, args.seed)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    write_manifest(args.output_dir / "pairs_manifest.csv", pairs)

    matcher = KeypointMatcher(
        device=device,
        max_keypoints=args.max_keypoints,
        max_side=args.max_side,
    )

    results = []
    for index, pair in enumerate(pairs, 1):
        pair_matches = matcher.count_matches(pair.before.path, pair.after.path)
        nonpair_matches = matcher.count_matches(pair.before.path, pair.negative.path)
        row = {
            "pair_id": index,
            "group_id": pair.group_id,
            "before": str(pair.before.path),
            "after": str(pair.after.path),
            "negative": str(pair.negative.path),
            "pair_matches": pair_matches,
            "nonpair_matches": nonpair_matches,
        }
        results.append(row)
        print(
            f"{index:03d}/{len(pairs)} "
            f"pair={pair_matches:4d} nonpair={nonpair_matches:4d} "
            f"{pair.before.path.name}",
            flush=True,
        )

    write_results(args.output_dir / "keypoint_counts.csv", results)
    write_summary(args.output_dir / "summary.json", results, args)
    plot_distribution(args.output_dir / "keypoint_distribution.png", results)

    print(f"\nSaved: {args.output_dir}")


def select_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_photos(dataset: Path, group_gap_seconds: int) -> list[Photo]:
    rows = []
    for path in dataset.glob("*.jpg"):
        match = FILENAME_RE.search(path.name)
        if match is None:
            continue
        date_part, time_part = match.groups()
        timestamp = datetime.strptime(date_part + time_part[:6], "%Y%m%d%H%M%S")
        annotation_path = path.with_suffix(".json")
        has_annotation = annotation_has_shapes(annotation_path)
        rows.append((timestamp, path, has_annotation))

    rows.sort()
    photos = []
    group_id = 0
    previous_timestamp = None
    for timestamp, path, has_annotation in rows:
        if (
            previous_timestamp is None
            or (timestamp - previous_timestamp).total_seconds() > group_gap_seconds
        ):
            group_id += 1
        photos.append(Photo(path, timestamp, has_annotation, group_id))
        previous_timestamp = timestamp

    return photos


def annotation_has_shapes(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return len(data.get("shapes", [])) > 0


def build_pairs(photos: list[Photo], seed: int) -> list[PhotoPair]:
    rng = random.Random(seed)
    negatives_by_group: dict[int, list[Photo]] = {}
    groups: dict[int, list[Photo]] = {}
    for photo in photos:
        groups.setdefault(photo.group_id, []).append(photo)
        if not photo.has_annotation:
            negatives_by_group.setdefault(photo.group_id, []).append(photo)

    pairs = []
    for group_id, group_photos in groups.items():
        index = 0
        while index < len(group_photos) - 1:
            current = group_photos[index]
            next_photo = group_photos[index + 1]
            if current.has_annotation and not next_photo.has_annotation:
                negative_candidates = [
                    photo
                    for candidate_group, negatives in negatives_by_group.items()
                    if candidate_group != group_id
                    for photo in negatives
                ]
                if negative_candidates:
                    pairs.append(
                        PhotoPair(
                            group_id=group_id,
                            before=current,
                            after=next_photo,
                            negative=rng.choice(negative_candidates),
                        ),
                    )
                index += 2
            else:
                index += 1

    return pairs


def write_manifest(path: Path, pairs: list[PhotoPair]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_id",
                "group_id",
                "before",
                "after",
                "negative",
                "before_time",
                "after_time",
                "negative_group_id",
            ],
        )
        writer.writeheader()
        for index, pair in enumerate(pairs, 1):
            writer.writerow(
                {
                    "pair_id": index,
                    "group_id": pair.group_id,
                    "before": pair.before.path,
                    "after": pair.after.path,
                    "negative": pair.negative.path,
                    "before_time": pair.before.timestamp.isoformat(sep=" "),
                    "after_time": pair.after.timestamp.isoformat(sep=" "),
                    "negative_group_id": pair.negative.group_id,
                },
            )


def write_results(path: Path, results: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def write_summary(path: Path, results: list[dict], args: argparse.Namespace) -> None:
    pair_counts = [row["pair_matches"] for row in results]
    nonpair_counts = [row["nonpair_matches"] for row in results]
    summary = {
        "dataset": str(args.dataset),
        "pairs_count": len(results),
        "group_gap_seconds": args.group_gap_seconds,
        "max_side": args.max_side,
        "max_keypoints": args.max_keypoints,
        "seed": args.seed,
        "pair": describe(pair_counts),
        "nonpair": describe(nonpair_counts),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def describe(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "min": min(values),
        "q05": percentile(sorted_values, 0.05),
        "q25": percentile(sorted_values, 0.25),
        "median": percentile(sorted_values, 0.50),
        "q75": percentile(sorted_values, 0.75),
        "q95": percentile(sorted_values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def plot_distribution(path: Path, results: list[dict]) -> None:
    pair_counts = [row["pair_matches"] for row in results]
    nonpair_counts = [row["nonpair_matches"] for row in results]

    plt.figure(figsize=(9, 5))
    plt.hist(pair_counts, bins=20, alpha=0.65, label="pairs")
    plt.hist(nonpair_counts, bins=20, alpha=0.65, label="non-pairs")
    plt.xlabel("Matched keypoints count")
    plt.ylabel("Image pairs")
    plt.title("LightGlue matched keypoints distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


class KeypointMatcher:
    def __init__(self, device: str, max_keypoints: int, max_side: int) -> None:
        self.device = device
        self.max_side = max_side
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    def count_matches(self, first: Path, second: Path) -> int:
        with TemporaryDirectory() as temp_dir:
            temp_first = Path(temp_dir) / "first.jpg"
            temp_second = Path(temp_dir) / "second.jpg"
            resize_for_matching(first, temp_first, self.max_side)
            resize_for_matching(second, temp_second, self.max_side)

            image0 = load_lightglue_image(temp_first).to(self.device)
            image1 = load_lightglue_image(temp_second).to(self.device)

            with torch.inference_mode():
                features0 = self.extractor.extract(image0)
                features1 = self.extractor.extract(image1)
                matches01 = self.matcher({"image0": features0, "image1": features1})

            matches01 = rbd(matches01)
            return int(matches01["matches"].shape[0])


def resize_for_matching(source: Path, destination: Path, max_side: int) -> None:
    image = Image.open(source)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side))
    image.save(destination, quality=92)


if __name__ == "__main__":
    main()
