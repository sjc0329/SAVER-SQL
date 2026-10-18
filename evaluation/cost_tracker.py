from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def usage_from_api(usage_obj: dict | None, prompt_text: str, completion_text: str, prices: dict) -> Usage:
    if usage_obj:
        prompt = int(usage_obj.get("prompt_tokens", 0) or 0)
        completion = int(usage_obj.get("completion_tokens", 0) or 0)
        total = int(usage_obj.get("total_tokens", prompt + completion) or 0)
    else:
        prompt = estimate_tokens(prompt_text)
        completion = estimate_tokens(completion_text)
        total = prompt + completion
    cost = (prompt / 1000.0) * float(prices.get("input_per_1k", 0.0))
    cost += (completion / 1000.0) * float(prices.get("output_per_1k", 0.0))
    return Usage(prompt, completion, total, cost)
