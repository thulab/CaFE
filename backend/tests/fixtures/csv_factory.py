from pathlib import Path


def write_csv(tmp_path: Path, content: str, name: str = "data.csv", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding=encoding)
    return path
