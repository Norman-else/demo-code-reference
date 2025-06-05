from .base import ReplenishmentAlgorithm

class ExponentialSmoothingAlgorithm(ReplenishmentAlgorithm):
    """Predicts demand using simple exponential smoothing."""

    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def predict(self, history):
        if not history:
            return 0
        forecast = history[0]
        for demand in history[1:]:
            forecast = self.alpha * demand + (1 - self.alpha) * forecast
        return forecast
