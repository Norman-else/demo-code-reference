from model import Vocabulary, PackagePredictor


def main():
    data = [
        ("cola", "500ml", "bottle"),
        ("orange juice", "1l", "bottle"),
        ("mineral water", "500ml", "bottle"),
        ("potato chips", "200g", "bag"),
        ("chocolate cookies", "300g", "bag"),
        ("rice 5kg", "5kg", "bag"),
        ("smartphone", "small", "box"),
        ("laptop computer", "medium", "box"),
        ("television", "large", "box"),
        ("breakfast cereal", "500g", "box"),
    ]

    names = [row[0] for row in data]
    sizes = [row[1] for row in data]
    types = [row[2] for row in data]

    vocab = Vocabulary()
    vocab.build(names)

    size_classes = sorted(set(sizes))
    type_classes = sorted(set(types))
    predictor = PackagePredictor(vocab, size_classes, type_classes)
    predictor.train(names, sizes, types)
    predictor.save("model.json")
    print("Model saved to model.json")


if __name__ == "__main__":
    main()
