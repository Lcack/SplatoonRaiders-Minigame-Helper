from dataclasses import asdict
import json
from pathlib import Path


def state_dict(state):
    return {"buttons": sorted(b.value for b in state.buttons),
            "left_stick": asdict(state.left_stick), "right_stick": asdict(state.right_stick)}


class TraceWriter:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")

    def write(self, obs, decision, executed=False):
        data = {"observation": asdict(obs), "decision": {"state": state_dict(decision.state),
                "reason": decision.reason, "confidence": decision.confidence,
                "diagnostics": decision.diagnostics}, "executed": executed}
        self.file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
