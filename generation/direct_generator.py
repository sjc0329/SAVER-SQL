from dataclasses import dataclass

from evaluation.cost_tracker import Usage
from llm.api_client import LLMClient
from llm.prompt_templates import (
    DECOMPOSITION_SYSTEM,
    DIRECT_SAMPLE_SYSTEM,
    DIRECT_SYSTEM,
    DIRECT_USER,
    EXECUTION_REPAIR_SYSTEM,
    EXECUTION_REPAIR_USER,
    QUERY_PLAN_SYSTEM,
)
from llm.response_parser import extract_sql


@dataclass
class Candidate:
    candidate_id: str
    generation_mode: str
    sql: str
    usage: Usage


class DirectGenerator:
    MODES = ["direct", "direct_sample", "decomposition", "query_plan"]

    def __init__(self, client: LLMClient, temperatures: dict[str, float], max_tokens: int | None = None):
        self.client = client
        self.temperatures = temperatures
        self.max_tokens = max_tokens

    def generate(self, mode: str, schema: str, question: str, evidence: str, examples: str = "") -> Candidate:
        system = {
            "direct": DIRECT_SYSTEM,
            "direct_sample": DIRECT_SAMPLE_SYSTEM,
            "decomposition": DECOMPOSITION_SYSTEM,
            "query_plan": QUERY_PLAN_SYSTEM,
        }[mode]
        response = self.client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": DIRECT_USER.format(schema=schema, question=question, evidence=evidence, examples=examples)},
            ],
            temperature=float(self.temperatures.get(mode, 0.0)),
            max_tokens=self.max_tokens,
        )
        candidate_id = f"c{self.MODES.index(mode) + 1}" if mode in self.MODES else f"c_{mode}"
        return Candidate(candidate_id, mode, extract_sql(response.text), response.usage)

    def repair_execution_error(self, schema: str, question: str, evidence: str, sql: str, error: str) -> Candidate:
        mode = "execution_repair"
        response = self.client.chat(
            [
                {"role": "system", "content": EXECUTION_REPAIR_SYSTEM},
                {
                    "role": "user",
                    "content": EXECUTION_REPAIR_USER.format(
                        schema=schema,
                        question=question,
                        evidence=evidence,
                        sql=sql,
                        error=error,
                    ),
                },
            ],
            temperature=float(self.temperatures.get(mode, 0.0)),
            max_tokens=self.max_tokens,
        )
        return Candidate("c5", mode, extract_sql(response.text), response.usage)
