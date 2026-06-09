from pathlib import Path

import pymupdf

COVER_WIDTH = 640


def generate_cover(pdf_path: Path, output_path: Path) -> None:
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        zoom = COVER_WIDTH / (page.rect.width or COVER_WIDTH)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(output_path, output="jpg", jpg_quality=82)
