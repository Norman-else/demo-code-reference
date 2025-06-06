import sys
from model import PackagePredictor


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <product name>")
        return
    name = " ".join(sys.argv[1:])
    predictor = PackagePredictor.load("model.json")
    (size_label, size_conf), (type_label, type_conf) = predictor.predict(name)
    print(f"Package size: {size_label} (confidence {size_conf:.2f})")
    print(f"Package type: {type_label} (confidence {type_conf:.2f})")


if __name__ == "__main__":
    main()
