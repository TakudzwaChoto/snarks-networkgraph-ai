import hashlib
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional


class AuditLog:
    """Hash-chained, append-only JSONL audit log.

    Each appended entry includes:
      - ts: ISO timestamp
      - type: event type string
      - payload: arbitrary JSON-serializable dict (summarized content)
      - prev: previous head hash (hex)
      - head: new head hash (hex)
    The head is computed as SHA256(prev || canonical_json(entry_without_head)).
    """

    def __init__(self, log_path: str = "logs/audit_log.jsonl", head_path: str = "logs/audit_head.txt") -> None:
        self.log_path = log_path
        self.head_path = head_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._lock = threading.Lock()
        # Initialize head if missing
        if not os.path.exists(self.head_path):
            with open(self.head_path, "w", encoding="utf-8") as f:
                f.write("".encode("utf-8").hex())

    @staticmethod
    def _canonical_json(d: Dict[str, Any]) -> str:
        return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _read_head(self) -> str:
        try:
            with open(self.head_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
        return "".encode("utf-8").hex()

    def _write_head(self, head_hex: str) -> None:
        with open(self.head_path, "w", encoding="utf-8") as f:
            f.write(head_hex)

    def append(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        with self._lock:
            prev = self._read_head()
            entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "type": event_type,
                "payload": payload,
                "prev": prev,
            }
            blob = (bytes.fromhex(prev) + self._canonical_json(entry).encode("utf-8"))
            head = hashlib.sha256(blob).hexdigest()
            entry["head"] = head
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._write_head(head)
            return entry

    def head(self) -> str:
        return self._read_head()