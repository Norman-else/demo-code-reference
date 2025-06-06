import csv
import hashlib
from pathlib import Path
from typing import List, Tuple

from model import Vocabulary, PackagePredictor


def load_data(csv_path: str) -> List[Tuple[str, str]]:
    """Load name and description pairs from a CSV file."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((row["name"], row["description"]))
    return records


def main():
    # Load data from a CSV file instead of hardcoding it
    data = load_data(Path(__file__).with_name("data.csv"))

    names = [row[0] for row in data]
    descriptions = [row[1] for row in data]

    # Since the CSV does not contain size/type labels, generate synthetic ones
    size_options = ["small", "medium", "large"]
    type_options = ["bag", "box", "bottle"]
    sizes = []
    types = []
    for name in names:
        digest = hashlib.md5(name.encode("utf-8")).digest()
        sizes.append(size_options[digest[0] % len(size_options)])
        types.append(type_options[digest[1] % len(type_options)])

    vocab = Vocabulary()
    vocab.build(names + descriptions)

    size_classes = sorted(set(sizes))
    type_classes = sorted(set(types))
    predictor = PackagePredictor(vocab, size_classes, type_classes)
    predictor.train(names, descriptions, sizes, types)
    predictor.save("model.json")
    print("Model saved to model.json")


if __name__ == "__main__":
    main()
