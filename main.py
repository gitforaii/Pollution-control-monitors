import json
import random
import sys
from typing import Any, Dict, List

import numpy as np

# Set global random seeds for reproducibility
random.seed(42)
np.random.seed(42)


class GridEnvironment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.pollution = np.zeros((height, width), dtype=float)

    def add_emission(self, x: int, y: int, amount: float) -> None:
        # Safely add or remove pollution from a cell
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pollution[y, x] += max(amount, 0.0)

    def apply_decay(self, decay_rate: float) -> None:
        # Simple global decay to prevent unbounded growth
        self.pollution *= max(0.0, 1.0 - decay_rate)
        np.clip(self.pollution, 0.0, None, out=self.pollution)

    def get_cell(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.pollution[y, x])
        return 0.0


class PollutionSource:
    def __init__(self, x: int, y: int, base_rate: float):
        self.x = x
        self.y = y
        self.base_rate = base_rate

    def step(self, env: "GridEnvironment") -> None:
        # Emit pollution with a small amount of randomness
        noise = np.random.uniform(0.8, 1.2)
        env.add_emission(self.x, self.y, self.base_rate * noise)


class MonitoringAgent:
    def __init__(self, x: int, y: int, threshold: float, mitigation_strength: float):
        self.x = x
        self.y = y
        self.threshold = threshold
        self.mitigation_strength = mitigation_strength

    def _move(self, width: int, height: int) -> None:
        # Simple random walk on the grid
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])
        nx = min(max(self.x + dx, 0), width - 1)
        ny = min(max(self.y + dy, 0), height - 1)
        self.x, self.y = nx, ny

    def step(self, env: "GridEnvironment") -> Dict[str, float]:
        # Move, monitor current cell, and mitigate if above threshold
        self._move(env.width, env.height)
        level = env.get_cell(self.x, self.y)
        alert = 0
        mitigated = 0.0
        if level > self.threshold:
            alert = 1
            reduction = level * self.mitigation_strength
            mitigated = reduction
            # Negative emission represents mitigation
            env.add_emission(self.x, self.y, -reduction)
        return {"alert": float(alert), "mitigated": float(mitigated)}


class SimulationController:
    def __init__(
        self,
        width: int,
        height: int,
        num_sources: int,
        num_agents: int,
        steps: int,
    ):
        self.env = GridEnvironment(width, height)
        self.sources = self._create_sources(num_sources, width, height)
        self.agents = self._create_agents(num_agents, width, height)
        self.steps = steps

    def _create_sources(self, count: int, width: int, height: int) -> List[PollutionSource]:
        sources: List[PollutionSource] = []
        for _ in range(count):
            x = random.randrange(width)
            y = random.randrange(height)
            rate = random.uniform(5.0, 15.0)
            sources.append(PollutionSource(x, y, rate))
        return sources

    def _create_agents(self, count: int, width: int, height: int) -> List[MonitoringAgent]:
        agents: List[MonitoringAgent] = []
        for _ in range(count):
            x = random.randrange(width)
            y = random.randrange(height)
            threshold = random.uniform(20.0, 40.0)
            agents.append(MonitoringAgent(x, y, threshold, mitigation_strength=0.5))
        return agents

    def _compute_metrics(self, step_index: int, alerts: int, mitigated: float) -> Dict[str, Any]:
        # Aggregate metrics for this step for later analysis
        total = float(self.env.pollution.sum())
        mean = float(self.env.pollution.mean())
        max_val = float(self.env.pollution.max())
        return {
            "step": step_index,
            "total_pollution": total,
            "mean_pollution": mean,
            "max_pollution": max_val,
            "alerts": float(alerts),
            "mitigated_amount": float(mitigated),
        }

    def run(self) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        for step_idx in range(self.steps):
            step_alerts = 0
            step_mitigated = 0.0
            for source in self.sources:
                source.step(self.env)
            for agent in self.agents:
                stats = agent.step(self.env)
                step_alerts += int(stats["alert"])
                step_mitigated += float(stats["mitigated"])
            self.env.apply_decay(decay_rate=0.03)
            metrics = self._compute_metrics(step_idx, step_alerts, step_mitigated)
            history.append(metrics)
            if step_idx % 20 == 0:
                # Print periodic status updates
                print(
                    f"Step {step_idx}: total={metrics['total_pollution']:.2f}, "
                    f"max={metrics['max_pollution']:.2f}, alerts={metrics['alerts']}"
                )
        totals = [h["total_pollution"] for h in history]
        avg_total = float(np.mean(totals))
        max_total = float(np.max(totals))
        print("Simulation finished.")
        print(f"Average total pollution: {avg_total:.2f}")
        print(f"Peak total pollution: {max_total:.2f}")
        return {
            "config": {
                "width": self.env.width,
                "height": self.env.height,
                "steps": self.steps,
                "num_sources": len(self.sources),
                "num_agents": len(self.agents),
            },
            "history": history,
        }


def save_history(data: Dict[str, Any], output_path: str) -> None:
    # Persist simulation results for visualization and ML
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved simulation history to {output_path}")


def main() -> None:
    # Basic CLI: python main.py [output_file]
    output_file = "simulation_history.json"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    controller = SimulationController(
        width=10,
        height=10,
        num_sources=4,
        num_agents=6,
        steps=200,
    )
    data = controller.run()
    save_history(data, output_file)


if __name__ == "__main__":
    main()
