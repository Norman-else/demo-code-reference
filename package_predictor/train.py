import csv
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

    # Since the CSV does not contain size/type labels, simulate them here
    label_map = {
        "cola": ("500ml", "bottle"),
        "orange juice": ("1l", "bottle"),
        "mineral water": ("500ml", "bottle"),
        "potato chips": ("200g", "bag"),
        "chocolate cookies": ("300g", "bag"),
        "rice 5kg": ("5kg", "bag"),
        "smartphone": ("small", "box"),
        "laptop computer": ("medium", "box"),
        "television": ("large", "box"),
        "breakfast cereal": ("500g", "box"),
    }

    sizes = [label_map[name][0] for name in names]
    types = [label_map[name][1] for name in names]

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
