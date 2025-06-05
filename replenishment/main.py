from .algorithms.naive import NaiveAlgorithm
from .algorithms.moving_average import MovingAverageAlgorithm
from .algorithms.exponential_smoothing import ExponentialSmoothingAlgorithm
from .evaluation import run_simulation
from .plotting import svg_bar_chart
from .simulation import generate_demand


def main() -> None:
    demand = generate_demand(days=30, products=2)

    algorithms = {
        "naive": NaiveAlgorithm(safety_stock=5, max_stock=40),
        "moving_avg": MovingAverageAlgorithm(window=3, safety_stock=5, max_stock=40),
        "exp_smooth": ExponentialSmoothingAlgorithm(alpha=0.5, safety_stock=5, max_stock=40),
    }

    metrics = run_simulation(demand, algorithms)
    service_levels = {
        name: round(m["service_level"] * 100, 2) for name, m in metrics.items()
    }

    svg_bar_chart(service_levels, "service_level.svg")

    for name, m in metrics.items():
        print(
            f"{name}: service_level={m['service_level']:.2f} avg_inventory={m['avg_inventory']:.2f}"
        )


if __name__ == "__main__":
    main()
