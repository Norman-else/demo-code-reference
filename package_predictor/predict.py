import sys
from model import PackagePredictor


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <product name> <description>")
        return
    name = sys.argv[1]
    description = " ".join(sys.argv[2:])
    predictor = PackagePredictor.load("model.json")
    (size_label, size_conf), (type_label, type_conf) = predictor.predict(name, description)
    print(f"Package size: {size_label} (confidence {size_conf:.2f})")
    print(f"Package type: {type_label} (confidence {type_conf:.2f})")


if __name__ == "__main__":
    main()
