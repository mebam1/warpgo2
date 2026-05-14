from pathlib import Path
from pprint import pprint

import torch


def main() -> None:
    mebam_dir = Path(__file__).resolve().parents[1]
    data_path = mebam_dir / "data" / "0.pt"

    obj = torch.load(data_path, map_location="cpu", weights_only=False)

    print(f"Loaded: {data_path}")
    print(f"Type: {type(obj)}")
    print("Contents:")
    pprint(obj)


if __name__ == "__main__":
    main()
