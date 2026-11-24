"""Test which chip trains fastest and correctly (GPU, TPU, CPU), in its own process, and print the choice
as JSON on the last line. Running it apart from the notebook matters on a TPU: a TPU can be held by one
process at a time, and a test inside the notebook would keep it from every script started afterwards."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heat.device import self_test  # noqa: E402

if __name__ == "__main__":
    best, _ = self_test(("bilstm", "resnet1d", "transformer"))
    print(json.dumps(best))
