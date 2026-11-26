"""Research-specific provenance adapter around PyAlex and EvoSkills paper tools."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .models import LiteratureScreen
from .storage import Evidence, atomic, canonical, digest


def normalized_query(value: str) -> str:
    return " ".join(value.casefold().split())


def normalized_text(value: str) -> str:
    return " ".join(value.split())


def public_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        return False
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return "." in host


def abstract_locations(urls):
    """Infer canonical arXiv abstract pages from OpenAlex PDF links."""
    for url in urls:
        parsed = urlparse(url)
        if parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} and parsed.path.startswith("/pdf/"):
            yield "https://arxiv.org/abs/" + parsed.path.removeprefix("/pdf/").removesuffix(".pdf")


def tool_environment() -> dict:
    # Public discovery tools do not receive the user's model/service credentials.
    return {k: v for k, v in os.environ.items()
            if not k.endswith(("_API_KEY", "_SECRET", "_TOKEN")) and not k.startswith("CODEX_")}


class LiteratureRadar:
    def __init__(self, root: Path, evidence: Evidence):
        self.root, self.evidence = root / "literature", evidence
        self.root.mkdir(parents=True, exist_ok=True)

    def history(self) -> list[str]:
        return [json.loads(path.read_bytes())["query"]
                for path in sorted((self.root / "searches").glob("*.json"))]

    def _search(self, query: str, limit=4) -> dict:
        key = digest(canonical({"query": normalized_query(query), "limit": limit, "provider": "OpenAlex/pyalex"}))
        path = self.root / "searches" / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_bytes())
            self.evidence.verify_tree(cached["evidence"])
            return {**cached, "cache_hit": True}
        request = {"query": query, "limit": limit}
        result = {"query": query, "provider": "OpenAlex/pyalex", "fetched": time.time(), "records": [], "errors": []}
        try:
            process = subprocess.run([sys.executable, "-m", "loop_research.literature_backend"], input=canonical(request),
                                     capture_output=True, timeout=35, env=tool_environment(), check=False)
            atomic(self.root / "logs" / f"{key}.stdout", process.stdout)
            atomic(self.root / "logs" / f"{key}.stderr", process.stderr)
            result["exit_code"] = process.returncode
            if process.returncode:
                result["errors"].append("OpenAlex client failed; inspect saved stderr")
            else:
                result["records"] = json.loads(process.stdout)["records"]
        except (subprocess.TimeoutExpired, ValueError, KeyError) as exc:
            result["errors"].append(type(exc).__name__)
        artifact = self.evidence.put("paper_search", result)
        cached = {"query": query, "evidence": artifact}
        atomic(path, canonical(cached))
        return {**cached, "cache_hit": False}

    def _read(self, record: dict, search_id: str) -> str:
        # DOI is the primary dedup key; repository IDs are used when no DOI exists.
        identity = (record.get("doi") or record["id"]).strip().casefold()
        key = digest(identity.encode())
        cache = self.root / "papers" / key / "source.json"
        location = record.get("best_oa_location") or record.get("primary_location") or {}
        locations = [location] + [item for item in record.get("locations", []) if item.get("is_oa")]
        urls = [item.get("pdf_url") for item in locations] + [item.get("landing_page_url") for item in locations] + [record.get("doi")]
        urls = list(dict.fromkeys(url for url in urls if url and public_url(url)))
        landings = list(dict.fromkeys([*abstract_locations(urls),
                                     *(url for url in urls if url not in {item.get("pdf_url") for item in locations})]))
        previous, attempted, attempted_landings = None, [], []
        if cache.exists():
            previous = json.loads(cache.read_bytes())["evidence"]
            self.evidence.verify_tree(previous)
            old = self.evidence.get(previous)["content"]
            if old["access"] in {"body_excerpt", "abstract_page"}:
                return previous
            attempted = old.get("attempted_urls", [old.get("url")])
            attempted_landings = old.get("attempted_abstract_urls", [])
            if not (any(url not in attempted for url in urls) or any(url not in attempted_landings for url in landings)):
                return previous
        source = {"identity": identity, "title": record.get("title") or "", "doi": record.get("doi"),
                  "year": record.get("publication_year"), "type": record.get("type"),
                  "authors": [a.get("author", {}).get("display_name") for a in record.get("authorships", [])[:12]],
                  "venue": (location.get("source") or {}).get("display_name"),
                  "retracted": record.get("is_retracted", False), "citations": record.get("cited_by_count", 0),
                  "abstract": record.get("abstract_text") or "", "urls": urls, "url": urls[0] if urls else None,
                  "body_excerpt": "", "body_evidence": None, "access": "metadata_only", "errors": [],
                  "trust": "external_unverified", "metadata": record, "attempted_urls": list(attempted),
                  "attempted_abstract_urls": list(attempted_landings)}
        parents = [search_id] + ([previous] if previous else [])
        script = Path(__file__).parent / "_vendor/evoskills/paper-navigator/scripts/fetch_paper.py"
        # Reuse the existing full-text tool. Outer timeout bounds its upstream retry policy.
        if urls and not source["retracted"]:
            for url in [url for url in urls if url not in attempted][:2]:
                folder = cache.parent / ("read-" + digest(url.encode())[:16])
                folder.mkdir(parents=True, exist_ok=True)
                source["attempted_urls"].append(url)
                command = [sys.executable, str(script), "--url", url, "--skip-metadata", "--papers-dir", str(folder),
                           "--json", "--limit-chars", "80000"]
                try:
                    process = subprocess.run(command, capture_output=True, timeout=45, env=tool_environment(), check=False)
                    atomic(folder / "reader.stdout", process.stdout)
                    atomic(folder / "reader.stderr", process.stderr)
                    if process.returncode:
                        source["errors"].append("Existing paper reader failed")
                    else:
                        output = json.loads(process.stdout)
                        body_path = Path(output["full_text_path"]).resolve()
                        body_path.relative_to(folder.resolve())
                        text = body_path.read_text()
                        # Access/error pages are not treated as paper bodies merely because HTTP succeeded.
                        blocked = re.search(r"access denied|captcha|just a moment|verify you are human|enable javascript and cookies", text[:1500], re.I)
                        title_words = {w.casefold() for w in re.findall(r"[A-Za-z]{4,}", source["title"])}
                        matching = sum(word in text.casefold() for word in title_words)
                        if len(text) >= 1500 and not blocked and matching >= min(3, len(title_words)) and title_words:
                            body = self.evidence.put("paper_body", {"url": url, "text": text,
                                "sha256": digest(text.encode()), "reader": "EvoSkills/fetch_paper", "fetched": time.time()})
                            parents.append(body)
                            source.update(body_evidence=body, body_excerpt=text[:10000], access="body_excerpt")
                        else:
                            source["errors"].append("Reader output did not establish accessible paper content")
                except (subprocess.TimeoutExpired, ValueError, OSError, KeyError) as exc:
                    source["errors"].append(type(exc).__name__)
                if source["access"] == "body_excerpt":
                    source["url"] = url
                    break
        # An accessible abstract page is sufficient for an abstract-level direction,
        # but cannot support mathematical or full-text claims.
        if source["access"] == "metadata_only" and not source["retracted"]:
            # The full-text reader and direct HTTP have different access paths.
            for landing in [url for url in landings if url not in attempted_landings][:3]:
                source["attempted_abstract_urls"].append(landing)
                try:
                    def check_redirect(request):
                        if not public_url(str(request.url)):
                            raise ValueError("Non-public paper redirect")
                    with httpx.Client(timeout=12, follow_redirects=True, event_hooks={"request": [check_redirect]}) as client:
                        response = client.get(landing)
                    document = BeautifulSoup(response.text, "html.parser")
                    text = document.get_text(" ", strip=True)
                    words = {w.casefold() for w in re.findall(r"[A-Za-z]{4,}", source["title"])}
                    if not response.is_success or not words or sum(w in text.casefold() for w in words) < min(3, len(words)):
                        continue
                    abstract_tag = document.select_one("meta[name='citation_abstract'], blockquote.abstract")
                    if abstract_tag:
                        abstract = abstract_tag.get("content") or abstract_tag.get_text(" ", strip=True)
                        abstract = re.sub(r"^Abstract\s*:\s*", "", abstract, flags=re.I)
                    else:
                        abstract = source["abstract"]
                        if normalized_text(abstract) not in normalized_text(text):
                            continue
                    if len(normalized_text(abstract)) >= 100:
                        source.update(abstract=abstract, url=str(response.url), access="abstract_page",
                                      landing_page_sha256=digest(response.content))
                        break
                except (httpx.HTTPError, ValueError) as exc:
                    source["errors"].append(type(exc).__name__)
        artifact = self.evidence.put("paper_source", source, parents)
        atomic(cache, canonical({"evidence": artifact}))
        return artifact

    def search(self, queries: list[str], *, max_papers=8) -> dict:
        searches = [self._search(query) for query in queries]
        records = {}
        groups = [[(record, search["evidence"]) for record in self.evidence.get(search["evidence"])["content"]["records"]]
                  for search in searches]
        # Preserve query diversity when the reading budget is smaller than the hit pool.
        for row in zip_longest(*groups):
            for pair in row:
                if pair is None:
                    continue
                record, search_id = pair
                identity = (record.get("doi") or record["id"]).casefold().strip()
                if identity not in records:
                    records[identity] = (record, search_id)
        # Read a bounded number; retain all raw results so truncation is visible.
        selected = list(records.values())[:max_papers]
        with ThreadPoolExecutor(max_workers=2) as pool:
            source_ids = list(pool.map(lambda pair: self._read(*pair), selected))
        return {"searches": searches, "source_ids": list(dict.fromkeys(source_ids)),
                "unique_hits": len(records), "unread_hits": max(0, len(records)-max_papers)}

    def cards(self, source_ids: list[str]) -> list[dict]:
        return [{"source_id": key, **{field: value for field, value in self.evidence.get(key)["content"].items()
                 if field not in {"metadata", "authors"}}} for key in source_ids]


def validate_screen(screen: LiteratureScreen, source_ids: list[str], allowed_targets: list[str], evidence: Evidence,
                    *, classifier_names: set[str] | None = None) -> list[dict]:
    """Source identity, access and exact quoted spans are code-enforced."""
    directions, seen, used = [], set(), set()
    for direction in screen.directions:
        if direction.machine_target not in allowed_targets:
            raise ValueError("Direction outside permitted intervention targets")
        if direction.machine_target == "classifier.family" and (not direction.classifier_name
                or direction.classifier_name not in (classifier_names or set())):
            raise ValueError("Classifier-family direction requires an available concrete classifier")
        for quote in direction.quotes:
            if quote.source_id not in source_ids:
                raise ValueError("Invented paper source")
            paper = evidence.get(quote.source_id)
            if paper["kind"] != "paper_source" or paper["content"]["retracted"]:
                raise ValueError("Invalid or retracted paper")
            source = paper["content"]
            if source["access"] not in {"body_excerpt", "abstract_page"}:
                raise ValueError("Inaccessible paper cannot support an executable direction")
            text = source["abstract"] if quote.field == "abstract" else source["body_excerpt"]
            if normalized_text(quote.quote) not in normalized_text(text):
                raise ValueError("Paper quotation not present in supplied original text")
            if quote.field == "body" and source["access"] != "body_excerpt":
                raise ValueError("Abstract-only access cannot support a full-text quotation")
            used.add(quote.source_id)
        value = direction.model_dump()
        key = digest(canonical(value))
        if key not in seen:
            directions.append({"id": key, **value, "trust": "screened_hypothesis_not_validated_fact"})
            seen.add(key)
    rejected = {item.source_id for item in screen.excluded_sources}
    if not rejected <= set(source_ids) or rejected & used or used | rejected != set(source_ids):
        raise ValueError("Every retrieved source needs an explicit use or exclusion decision")
    return sorted(directions, key=lambda d: (-d["priority"], d["id"]))
