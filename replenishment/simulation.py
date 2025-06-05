import random
from typing import Dict, List

def generate_demand(days: int, products: int, seed: int = 42) -> Dict[str, List[int]]:
    """Generate simulated daily demand for multiple products."""
    random.seed(seed)
    demand = {}
    for p in range(products):
        base = random.randint(5, 15)
        demand[f"product_{p+1}"] = [max(0, base + random.randint(-3, 3)) for _ in range(days)]
    return demand
