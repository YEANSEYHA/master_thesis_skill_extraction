import os

from rich.console import Console

console = Console()


def read_all_htmls(folder_path: str, limit: int = None, offset: int = 0) -> list[dict]:
    all_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".html"))
    total_available = len(all_files)

    # Slice by offset and limit
    end = offset + limit if limit else None
    selected = all_files[offset:end]

    results = []
    for filename in selected:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        results.append({"filename": filename, "content": content})

    console.print(f"[bold]Total HTML files:[/bold] {len(results)} (range {offset}–{offset + len(results) - 1} of {total_available})")
    return results
