from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _to_rgb(value: float) -> tuple[int, int, int]:
    value = float(min(1.0, max(0.0, value)))
    stops = [
        (0.0, (15, 23, 42)),
        (0.35, (36, 99, 235)),
        (0.7, (56, 189, 248)),
        (1.0, (250, 204, 21)),
    ]
    for index in range(len(stops) - 1):
        left_pos, left_color = stops[index]
        right_pos, right_color = stops[index + 1]
        if value <= right_pos:
            ratio = 0.0 if right_pos == left_pos else (value - left_pos) / (right_pos - left_pos)
            return tuple(
                int(round(left_color[channel] + ratio * (right_color[channel] - left_color[channel])))
                for channel in range(3)
            )
    return stops[-1][1]


def _attention_to_image(attention_map: np.ndarray, cell_size: int) -> Image.Image:
    seq_len = int(attention_map.shape[0])
    image = Image.new("RGB", (seq_len * cell_size, seq_len * cell_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(seq_len):
        for col in range(seq_len):
            color = _to_rgb(float(attention_map[row, col]))
            draw.rectangle(
                (
                    col * cell_size,
                    row * cell_size,
                    (col + 1) * cell_size - 1,
                    (row + 1) * cell_size - 1,
                ),
                fill=color,
            )
    return image


def save_attention_maps_image(
    attention_maps: list,
    output_path: str | Path,
    title: str,
    source_name: str,
    sample_index: int = 0,
    cell_size: int = 18,
    padding: int = 16,
    header_height: int = 72,
) -> Path | None:
    sample_maps = []
    for attention_map in attention_maps:
        if attention_map is None:
            continue
        if sample_index < 0 or sample_index >= attention_map.shape[0]:
            continue
        sample_maps.append(attention_map[sample_index].float().numpy())

    if not sample_maps:
        return None

    num_heads = int(sample_maps[0].shape[0])
    seq_len = int(sample_maps[0].shape[-1])
    panel_size = seq_len * cell_size
    grid_cols = min(4, num_heads)
    grid_rows = int(np.ceil(num_heads / grid_cols))
    label_height = 18
    layer_title_height = 20
    width = padding * (grid_cols + 1) + panel_size * grid_cols
    layer_height = layer_title_height + grid_rows * (panel_size + label_height + padding)
    height = header_height + len(sample_maps) * layer_height + padding

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((padding, 10), title, fill=(17, 24, 39))
    draw.text((padding, 30), source_name, fill=(75, 85, 99))
    draw.text((padding, 50), f"sample_index={sample_index}, per-head maps", fill=(75, 85, 99))

    y_offset = header_height
    for layer_index, attention_map in enumerate(sample_maps):
        draw.text((padding, y_offset), f"layer {layer_index}", fill=(17, 24, 39))
        y_offset += layer_title_height
        for head_index in range(num_heads):
            row = head_index // grid_cols
            col = head_index % grid_cols
            panel_x = padding + col * (panel_size + padding)
            panel_y = y_offset + row * (panel_size + label_height + padding)
            draw.text((panel_x, panel_y), f"head {head_index}", fill=(17, 24, 39))
            panel = _attention_to_image(attention_map[head_index], cell_size=cell_size)
            canvas.paste(panel, (panel_x, panel_y + label_height))
        y_offset += grid_rows * (panel_size + label_height + padding)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
