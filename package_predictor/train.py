import csv
from pathlib import Path

from model import Vocabulary, PackagePredictor


def load_data(csv_path: str):
    """Load training data from a CSV file."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                (row["name"], row["description"], row["size"], row["type"])
            )
    return records


def main():
    # Load data from a CSV file instead of hardcoding it
    data = load_data(Path(__file__).with_name("data.csv"))

    names = [row[0] for row in data]
    descriptions = [row[1] for row in data]
    sizes = [row[2] for row in data]
    types = [row[3] for row in data]

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
