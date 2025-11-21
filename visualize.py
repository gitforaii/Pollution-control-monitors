import json
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_history(path: str) -> Dict[str, Any]:
    # Load simulation history JSON file
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(history: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    # Convert history list into numeric series for plotting
    steps = np.array([h["step"] for h in history], dtype=float)
    total = np.array([h["total_pollution"] for h in history], dtype=float)
    mean = np.array([h["mean_pollution"] for h in history], dtype=float)
    max_vals = np.array([h["max_pollution"] for h in history], dtype=float)
    alerts = np.array([h["alerts"] for h in history], dtype=float)
    mitigated = np.array([h["mitigated_amount"] for h in history], dtype=float)
    return {
        "steps": steps,
        "total": total,
        "mean": mean,
        "max": max_vals,
        "alerts": alerts,
        "mitigated": mitigated,
    }


def plot_pollution_levels(series: Dict[str, np.ndarray]) -> None:
    # Time-series of different pollution metrics
    plt.figure(figsize=(8, 4))
    plt.plot(series["steps"], series["total"], label="Total pollution")
    plt.plot(series["steps"], series["mean"], label="Mean pollution")
    plt.plot(series["steps"], series["max"], label="Max cell pollution")
    plt.xlabel("Step")
    plt.ylabel("Pollution level")
    plt.title("Pollution levels over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pollution_timeseries.png", dpi=150)
    print("Saved plot to pollution_timeseries.png")
    print("Close each window to see the next plot")
    plt.show()


def plot_alerts_and_mitigation(series: Dict[str, np.ndarray]) -> None:
    # Alerts and mitigation activity over time
    plt.figure(figsize=(8, 4))
    plt.bar(series["steps"], series["alerts"], label="Alerts", alpha=0.6)
    plt.plot(series["steps"], series["mitigated"], color="green", label="Mitigated amount")
    plt.xlabel("Step")
    plt.ylabel("Count / Amount")
    plt.title("Alerts and mitigation over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("alerts_mitigation.png", dpi=150)
    print("Saved plot to alerts_mitigation.png")
    print("Close each window to see the next plot")
    plt.show()


def plot_pollution_distribution(series: Dict[str, np.ndarray]) -> None:
    # Distribution of total pollution levels across the run
    plt.figure(figsize=(6, 4))
    plt.hist(series["total"], bins=20, color="steelblue", alpha=0.8)
    plt.xlabel("Total pollution level")
    plt.ylabel("Frequency")
    plt.title("Distribution of total pollution across steps")
    plt.tight_layout()
    plt.savefig("pollution_distribution.png", dpi=150)
    print("Saved plot to pollution_distribution.png")
    print("Close each window to see the next plot")
    plt.show()


def main() -> None:
    # Basic CLI: python visualize.py [history_file]
    input_file = "simulation_history.json"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    data = load_history(input_file)
    history = data.get("history", [])
    if not history:
        print(f"No history entries found in {input_file}")
        return
    series = extract_series(history)
    plot_pollution_levels(series)
    plot_alerts_and_mitigation(series)
    plot_pollution_distribution(series)


if __name__ == "__main__":
    main()
