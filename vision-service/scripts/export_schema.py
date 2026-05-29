"""Export the authoritative DetectionEvent JSON Schema for the SignalK plugin.

Run from the vision-service directory:
    python scripts/export_schema.py
Writes ../signalk-plugin/schema/detection-event.schema.json so the plugin can
validate inbound events against the exact contract the container emits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import DetectionEvent  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent.parent
       / "signalk-plugin" / "schema" / "detection-event.schema.json")


def main() -> None:
    schema = DetectionEvent.model_json_schema()
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = "DetectionEvent"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
