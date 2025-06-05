from .base import ReplenishmentAlgorithm

class NaiveAlgorithm(ReplenishmentAlgorithm):
    """Naive algorithm that assumes next demand equals last demand."""

    def predict(self, history):
        return history[-1] if history else 0
