from typing import Dict


def svg_bar_chart(data: Dict[str, float], filename: str) -> None:
    """Generate a simple bar chart as an SVG file."""
    width = 400
    height = 300
    margin = 40
    bar_width = (width - 2 * margin) / len(data)
    max_value = max(data.values()) if data else 1
    scale = (height - 2 * margin) / max_value

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    ]
    y_axis = height - margin
    for idx, (label, value) in enumerate(data.items()):
        x = margin + idx * bar_width
        bar_height = value * scale
        y = y_axis - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width * 0.8}" height="{bar_height}" fill="steelblue"/>')
        parts.append(f'<text x="{x + bar_width * 0.4}" y="{y_axis + 15}" text-anchor="middle" font-size="12">{label}</text>')
        parts.append(f'<text x="{x + bar_width * 0.4}" y="{y - 5}" text-anchor="middle" font-size="12">{value:.2f}</text>')
    parts.append('</svg>')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
