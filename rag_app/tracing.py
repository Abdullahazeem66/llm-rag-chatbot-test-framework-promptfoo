import time
from contextlib import contextmanager

from pydantic import BaseModel

# USD per 1M tokens: (input, output)
PRICES_PER_1M = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    aux_input_tokens: int = 0
    aux_output_tokens: int = 0
    cost_usd: float | None = 0.0


def price_for(model: str) -> tuple[float, float] | None:
    matches = [key for key in PRICES_PER_1M if model == key or model.startswith(key + "-")]
    return PRICES_PER_1M[max(matches, key=len)] if matches else None


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 8)


def add_costs(*costs: float | None) -> float | None:
    return None if any(c is None for c in costs) else round(sum(costs), 8)


class Timings:
    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def track(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[stage] = round((time.perf_counter() - start) * 1000, 1)
