from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image, ImageOps

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image as load_lightglue_image
from lightglue.utils import rbd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path


@dataclass(frozen=True)
class Triplet:
    pair_id: str
    before_path: Path
    after_path: Path
    negative_id: str
    negative_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-dir", type=Path, default=Path(r"C:\shared\CG\ЧИ\До"))
    parser.add_argument("--after-dir", type=Path, default=Path(r"C:\shared\CG\ЧИ\После"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tmatch"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "command",
        choices=("prepare", "count"),
        help="prepare writes a triplet manifest; count computes match counts from it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "tmatch_triplets_manifest.csv"
    results_path = args.output_dir / "tmatch_keypoint_counts.csv"

    if args.command == "prepare":
        before_images = index_images(args.before_dir)
        after_images = index_images(args.after_dir)
        triplets = build_triplets(before_images, after_images, seed=args.seed)
        if args.limit is not None:
            triplets = triplets[: args.limit]

        write_manifest(manifest_path, triplets)
        print_dataset_report(before_images, after_images, triplets)
        print(f"Manifest saved to: {manifest_path}")
        return

    triplets = read_manifest(manifest_path)
    if args.limit is not None:
        triplets = triplets[: args.limit]

    existing_rows = read_existing_results(results_path)
    completed_pair_ids = {row["pair_id"] for row in existing_rows}

    device = select_device(args.device)
    print_device_report(device)

    matcher = KeypointMatcher(
        device=device,
        max_keypoints=args.max_keypoints,
        max_side=args.max_side,
    )
    rows = existing_rows.copy()
    for row_number, triplet in enumerate(triplets, start=1):
        if triplet.pair_id in completed_pair_ids:
            print(f"{row_number}/{len(triplets)} id={triplet.pair_id} skipped", flush=True)
            continue

        positive_matches = matcher.count_matches(triplet.before_path, triplet.after_path)
        negative_matches = matcher.count_matches(triplet.before_path, triplet.negative_path)
        row = {
            "pair_id": triplet.pair_id,
            "before_path": str(triplet.before_path),
            "after_path": str(triplet.after_path),
            "negative_id": triplet.negative_id,
            "negative_path": str(triplet.negative_path),
            "positive_matches": positive_matches,
            "negative_matches": negative_matches,
        }
        rows.append(row)
        write_results(results_path, rows)
        print(
            f"{row_number}/{len(triplets)} "
            f"id={triplet.pair_id} pos={positive_matches} neg={negative_matches}",
            flush=True,
        )

    write_results(results_path, rows)
    print(f"Results saved to: {results_path}")


def index_images(directory: Path) -> dict[str, ImageRecord]:
    images: dict[str, ImageRecord] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_id = path.stem.strip()
        if image_id in images:
            raise ValueError(f"Duplicate image id {image_id!r}: {images[image_id].path} and {path}")
        images[image_id] = ImageRecord(image_id=image_id, path=path)
    return dict(sorted(images.items(), key=lambda item: sort_key(item[0])))


def sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def build_triplets(
    before_images: dict[str, ImageRecord],
    after_images: dict[str, ImageRecord],
    seed: int,
) -> list[Triplet]:
    rng = random.Random(seed)
    pair_ids = sorted(before_images.keys() & after_images.keys(), key=sort_key)
    if len(pair_ids) < 2:
        raise ValueError("Need at least two matched pairs to sample negative examples")

    triplets = []
    for pair_id in pair_ids:
        negative_candidates = [candidate_id for candidate_id in pair_ids if candidate_id != pair_id]
        negative_id = rng.choice(negative_candidates)
        triplets.append(
            Triplet(
                pair_id=pair_id,
                before_path=before_images[pair_id].path,
                after_path=after_images[pair_id].path,
                negative_id=negative_id,
                negative_path=after_images[negative_id].path,
            ),
        )
    return triplets


def print_dataset_report(
    before_images: dict[str, ImageRecord],
    after_images: dict[str, ImageRecord],
    triplets: list[Triplet],
) -> None:
    before_ids = set(before_images)
    after_ids = set(after_images)
    only_before = sorted(before_ids - after_ids, key=sort_key)
    only_after = sorted(after_ids - before_ids, key=sort_key)

    print(f"Before images: {len(before_images)}")
    print(f"After images: {len(after_images)}")
    print(f"Matched pairs: {len(before_ids & after_ids)}")
    print(f"Only before: {len(only_before)}")
    print(f"Only after: {len(only_after)}")
    print(f"Triplets: {len(triplets)}")
    if only_before[:10]:
        print(f"First only-before ids: {', '.join(only_before[:10])}")
    if only_after[:10]:
        print(f"First only-after ids: {', '.join(only_after[:10])}")


def write_manifest(path: Path, triplets: list[Triplet]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_id",
                "before_path",
                "after_path",
                "negative_id",
                "negative_path",
            ],
        )
        writer.writeheader()
        for triplet in triplets:
            writer.writerow(
                {
                    "pair_id": triplet.pair_id,
                    "before_path": triplet.before_path,
                    "after_path": triplet.after_path,
                    "negative_id": triplet.negative_id,
                    "negative_path": triplet.negative_path,
                },
            )


def read_manifest(path: Path) -> list[Triplet]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return [
            Triplet(
                pair_id=row["pair_id"],
                before_path=Path(row["before_path"]),
                after_path=Path(row["after_path"]),
                negative_id=row["negative_id"],
                negative_path=Path(row["negative_path"]),
            )
            for row in csv.DictReader(file)
        ]


def write_results(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_report(device: str) -> None:
    cuda_version = torch.version.cuda or "none"
    if device == "cuda" and torch.cuda.is_available():
        print(f"Device: cuda ({torch.cuda.get_device_name(0)}), torch CUDA: {cuda_version}")
        return
    print(f"Device: {device}, torch CUDA: {cuda_version}")


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
