"""Run the existing OpenAlex client in a time-bounded, credential-free process."""
from __future__ import annotations

import json
import sys

from pyalex import Works


def main():
    request = json.loads(sys.stdin.read())
    works = Works().search(request["query"]).filter(is_retracted=False).get(per_page=request["limit"])
    records = []
    for work in works:
        record = dict(work)
        record["abstract_text"] = work["abstract"] if work.get("abstract_inverted_index") else ""
        records.append(record)
    sys.stdout.write(json.dumps({"provider": "OpenAlex/pyalex", "records": records}, allow_nan=False))


if __name__ == "__main__":
    main()
