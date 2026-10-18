from pathlib import Path
from typing import Any

from data.bird_loader import load_json


class SchemaReader:
    def __init__(self, tables_file: str | Path):
        self.tables_file = Path(tables_file)
        self._schemas = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        entries = load_json(self.tables_file)
        return {entry["db_id"]: entry for entry in entries}

    def get(self, db_id: str) -> dict[str, Any]:
        if db_id not in self._schemas:
            raise KeyError(f"Unknown db_id in schema file: {db_id}")
        return self._schemas[db_id]

    def all_db_ids(self) -> list[str]:
        return sorted(self._schemas)
