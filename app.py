from __future__ import annotations

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from zero_shot_cleaning_assessment import ZeroShotCleaningAssessment


method = ZeroShotCleaningAssessment()
MAX_MATCH_LINES = 120
MATCH_COLORS = [
    "lime",
    "cyan",
    "yellow",
    "magenta",
    "orange",
    "red",
    "dodgerblue",
    "springgreen",
]


def remember_original(image):
    return image


def clear_original():
    return None


def assess_cleaning(image_before, image_after, original_before, original_after):
    source_before = original_before or image_before
    source_after = original_after or image_after

    if source_before is None or source_after is None:
        return (
            "Загрузите два изображения: до очистки и после очистки.",
            source_before,
            source_after,
            original_before,
            original_after,
        )

    result = method.assess(source_before, source_after)

    lines = [
        result.message,
        f"Совпадений ключевых точек: {result.matches_count}",
        f"Объектов до очистки: {result.objects_before}",
        f"Объектов после очистки: {result.objects_after}",
    ]

    if result.cleaning_score is not None:
        lines.append(f"Критерий качества очистки: {result.cleaning_score:.2f}%")

    before_with_overlay = draw_result_overlay(
        source_before,
        result.detections_before,
        result.matched_points,
        point_key="before",
        other_image_width=source_after.width,
    )
    after_with_overlay = draw_result_overlay(
        source_after,
        result.detections_after,
        result.matched_points,
        point_key="after",
        other_image_width=source_before.width,
    )

    return "\n".join(lines), before_with_overlay, after_with_overlay, source_before, source_after


def draw_result_overlay(
    image: Image.Image,
    detections: list[dict],
    matched_points: list[dict[str, list[float]]],
    point_key: str,
    other_image_width: int,
) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw_keypoint_lines(draw, matched_points, point_key, image.width, other_image_width)
    draw_detections(draw, detections, font)

    return image


def draw_keypoint_lines(
    draw: ImageDraw.ImageDraw,
    matched_points: list[dict[str, list[float]]],
    point_key: str,
    image_width: int,
    other_image_width: int,
) -> None:
    for index, point in enumerate(matched_points[:MAX_MATCH_LINES]):
        color = MATCH_COLORS[index % len(MATCH_COLORS)]
        x0, y0 = point["before"]
        x1, y1 = point["after"]
        x1_on_combined_canvas = x1 + image_width

        if point_key == "before":
            y_at_right_border = y_on_line(x0, y0, x1_on_combined_canvas, y1, image_width)
            draw.line((x0, y0, image_width, y_at_right_border), fill=color, width=2)
            draw.ellipse((x0 - 4, y0 - 4, x0 + 4, y0 + 4), fill=color, outline="black", width=1)
        else:
            x1_on_combined_canvas = other_image_width + x1
            y_at_left_border = y_on_line(x0, y0, x1_on_combined_canvas, y1, other_image_width)
            draw.line((0, y_at_left_border, x1, y1), fill=color, width=2)
            draw.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), fill=color, outline="black", width=1)


def y_on_line(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


def draw_detections(
    draw: ImageDraw.ImageDraw,
    detections: list[dict],
    font: ImageFont.ImageFont,
) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = str(detection["label"])
        score = detection["score"]
        text = f"{label} {score:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text_box = draw.textbbox((x1, y1), text, font=font)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        text_y = max(0, y1 - text_height - 4)

        draw.rectangle(
            (x1, text_y, x1 + text_width + 6, text_y + text_height + 4),
            fill="red",
        )
        draw.text((x1 + 3, text_y + 2), text, fill="white", font=font)


with gr.Blocks(title="Оценка качества очистки") as demo:
    gr.Markdown("## Оценка качества очистки территории")

    with gr.Row():
        image_before = gr.Image(type="pil", label="Изображение до очистки")
        image_after = gr.Image(type="pil", label="Изображение после очистки")

    original_before = gr.State()
    original_after = gr.State()

    check_button = gr.Button("Оценить")
    result_message = gr.Textbox(label="Результат", lines=6)

    image_before.upload(fn=remember_original, inputs=image_before, outputs=original_before)
    image_after.upload(fn=remember_original, inputs=image_after, outputs=original_after)
    image_before.clear(fn=clear_original, outputs=original_before)
    image_after.clear(fn=clear_original, outputs=original_after)

    check_button.click(
        fn=assess_cleaning,
        inputs=[image_before, image_after, original_before, original_after],
        outputs=[result_message, image_before, image_after, original_before, original_after],
    )


if __name__ == "__main__":
    demo.launch()
