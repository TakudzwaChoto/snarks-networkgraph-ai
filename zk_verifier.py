import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ZkEvent:
    timestamp_ms: int
    verified: bool
    reason: str
    verify_ms: float


class ZkMetrics:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._buffer: List[ZkEvent] = []
        # Load existing
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        evt = json.loads(line)
                        self._buffer.append(ZkEvent(**evt))
            except Exception:
                # start fresh on error
                self._buffer = []

    def record(self, event: ZkEvent) -> None:
        self._buffer.append(event)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def summary(self, last_n: int = 500) -> Dict[str, Any]:
        window = self._buffer[-last_n:] if last_n and len(self._buffer) > last_n else self._buffer
        total = len(window)
        ok = sum(1 for e in window if e.verified)
        fail = total - ok
        avg_ms = sum(e.verify_ms for e in window) / total if total else 0.0
        reasons: Dict[str, int] = {}
        for e in window:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        return {
            "total": total,
            "verified": ok,
            "failed": fail,
            "avg_verify_ms": round(avg_ms, 2),
            "failure_reasons": reasons,
        }

    def series(self, last_n: int = 200) -> Dict[str, List[Any]]:
        window = self._buffer[-last_n:] if last_n and len(self._buffer) > last_n else self._buffer
        return {
            "timestamps": [e.timestamp_ms for e in window],
            "verified": [1 if e.verified else 0 for e in window],
            "latency_ms": [e.verify_ms for e in window],
            "reasons": [e.reason for e in window],
        }


class ZkVerifier:
    def __init__(self, verifier_key_path: str = "zk/verifier_key.json", log_path: str = "logs/zk_metrics.jsonl") -> None:
        self.verifier_key_path = verifier_key_path
        self.metrics = ZkMetrics(log_path)

    def is_configured(self) -> bool:
        return os.path.exists(self.verifier_key_path)

    @staticmethod
    def _snarkjs_path() -> Optional[str]:
        path = shutil.which("snarkjs")
        return path

    def is_available(self) -> bool:
        return self.is_configured() and self._snarkjs_path() is not None

    def verify(self, proof: Dict[str, Any], public_signals: List[Any]) -> bool:
        start = time.time()
        reason = ""
        ok = False
        if not self.is_configured():
            reason = "verifier_key_missing"
            ok = False
        elif self._snarkjs_path() is None:
            reason = "snarkjs_not_found"
            ok = False
        else:
            # Write temp files and run `snarkjs groth16 verify <vk> public.json proof.json`
            with tempfile.TemporaryDirectory() as td:
                pub_path = os.path.join(td, "public.json")
                proof_path = os.path.join(td, "proof.json")
                with open(pub_path, "w", encoding="utf-8") as f:
                    json.dump(public_signals, f)
                with open(proof_path, "w", encoding="utf-8") as f:
                    json.dump(proof, f)
                try:
                    result = subprocess.run(
                        [self._snarkjs_path(), "groth16", "verify", self.verifier_key_path, pub_path, proof_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                        text=True,
                    )
                    ok = "OK!" in result.stdout
                    reason = "ok" if ok else (result.stderr.strip() or "verification_failed")
                except Exception as e:
                    ok = False
                    reason = f"exec_error:{e}"
        ms = (time.time() - start) * 1000.0
        self.metrics.record(ZkEvent(timestamp_ms=int(time.time() * 1000), verified=ok, reason=reason, verify_ms=ms))
        return ok

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "configured": self.is_configured(),
            "snarkjs": self._snarkjs_path() is not None,
            "summary": self.metrics.summary(),
            "series": self.metrics.series(),
        }