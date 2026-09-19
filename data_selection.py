from pathlib import Path
import re
import tqdm
import os

DATA_DIR = Path("D:\\Downloads\\Compressed\\data")
OUTPUT_DIR = Path(__file__).parent / "processed_data"


if __name__ == "__main__":
    wins = 0
    for folder in tqdm.tqdm(
        DATA_DIR.iterdir(), desc="Processing folders", unit="folder"
    ):
        if folder.is_dir():
            name = folder.name
            result = name.split("_")[-1]
            if result == "win":
                # Copy the folder to the output directory
                wins += 1
                new_path = OUTPUT_DIR / name
                new_path.mkdir(parents=True, exist_ok=True)
                for item in folder.iterdir():
                    if item.is_file():
                        new_item_path = new_path / item.name
                        if not new_item_path.exists():
                            os.link(item, new_item_path)

    print(f"Processed {wins} winning trajectories.")
