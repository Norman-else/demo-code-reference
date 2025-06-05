class ReplenishmentAlgorithm:
    """Base class for replenishment algorithms."""
    def __init__(self, safety_stock: int, max_stock: int):
        self.safety_stock = safety_stock
        self.max_stock = max_stock

    def predict(self, history: list) -> float:
        """Predict demand for next period based on demand history."""
        raise NotImplementedError

    def order_quantity(self, current_stock: int, history: list) -> int:
        """Return the order quantity for the current period."""
        prediction = self.predict(history)
        target = int(round(prediction)) + self.safety_stock
        return max(0, min(self.max_stock - current_stock, target - current_stock))
