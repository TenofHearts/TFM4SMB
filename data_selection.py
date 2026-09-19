from pathlib import Path
import re

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "processed_data"

regex = r"_(win|fail)$"

if __name__ == "__main__":
    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            name = folder.name
            match = re.search(regex, name)
            if match:
                result = match.group(1)
                if result == "win":
                    # Copy the folder to the output directory
                    new_path = OUTPUT_DIR / name
                    new_path.mkdir(parents=True, exist_ok=True)
                    for item in folder.iterdir():
                        if item.is_file():
                            new_item_path = new_path / item.name
                            new_item_path.write_bytes(item.read_bytes())
