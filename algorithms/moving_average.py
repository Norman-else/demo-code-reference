from .base import ReplenishmentAlgorithm

class MovingAverageAlgorithm(ReplenishmentAlgorithm):
    """Predicts demand using simple moving average."""

    def __init__(self, window: int, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def predict(self, history):
        if not history:
            return 0
        if len(history) < self.window:
            return sum(history) / len(history)
        return sum(history[-self.window:]) / self.window
