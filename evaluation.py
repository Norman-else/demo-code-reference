from typing import Dict, List

from algorithms.base import ReplenishmentAlgorithm

def run_simulation(demand: Dict[str, List[int]], algorithms: Dict[str, ReplenishmentAlgorithm], initial_stock: int = 20):
    """Run simulation for each algorithm and return metrics."""
    results = {name: {"stockouts": 0, "inventory_levels": []} for name in algorithms}
    days = len(next(iter(demand.values())))
    total_days = days * len(demand)
    for product_demands in demand.values():
        for name, alg in algorithms.items():
            stock = initial_stock
            history: List[int] = []
            for d in product_demands:
                order = alg.order_quantity(stock, history)
                stock += order
                if stock < d:
                    results[name]["stockouts"] += 1
                    stock = 0
                else:
                    stock -= d
                history.append(d)
                results[name]["inventory_levels"].append(stock)
    metrics = {}
    for name, r in results.items():
        service_level = 1 - r["stockouts"] / total_days
        avg_inventory = sum(r["inventory_levels"]) / len(r["inventory_levels"])
        metrics[name] = {
            "service_level": service_level,
            "avg_inventory": avg_inventory,
        }
    return metrics
