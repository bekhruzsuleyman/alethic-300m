"""Cost guard: speed, model FLOP utilization, projected time and dollars, and --max-cost."""
from __future__ import annotations

from dataclasses import dataclass


def mfu(tokens_per_sec: float, flops_per_token: float, peak_flops: float) -> float:
    """Model FLOP utilization: achieved FLOP/s divided by the GPU's peak FLOP/s."""
    return tokens_per_sec * flops_per_token / peak_flops


@dataclass
class CostGuard:
    """Turns wall-clock time into dollars at a fixed price per hour."""
    price_per_hour: float
    max_cost: float | None = None

    def spent(self, wall_seconds: float) -> float:
        """Dollars spent so far for this run (all resumes included)."""
        return wall_seconds / 3600.0 * self.price_per_hour

    def over_budget(self, wall_seconds: float) -> bool:
        """True once the spend has reached --max-cost."""
        return self.max_cost is not None and self.spent(wall_seconds) >= self.max_cost

    def projection(self, tokens_left: int, tokens_per_sec: float) -> tuple[float, float]:
        """(hours left, dollars left) at the current speed."""
        if tokens_per_sec <= 0:
            return float("inf"), float("inf")
        hours = tokens_left / tokens_per_sec / 3600.0
        return hours, hours * self.price_per_hour
