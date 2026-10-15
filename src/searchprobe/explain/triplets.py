"""(query, satisfying document, violating document) triplets.

A retriever "understands" an operator for a case if it scores a document that
satisfies the query above a near-duplicate that violates it. Triplets come
from two sources:

- **Known sets**: Wikipedia summaries of an answer-set entity (d+) and a
  negative-set entity (d-) of a Wikidata-backed case (Norway vs France for
  "European countries not in the EU"). No judging needed.
- **Judged pools**: a fully satisfying and a violating document that search
  engines actually returned for the case, as decided by the per-document judge.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from searchprobe.cases.models import KnownEntity, TestCase
from searchprobe.judging.checkers import wikipedia_title


@dataclass
class Triplet:
    case_id: str
    category: str
    query: str
    positive: str
    negative: str
    positive_id: str
    negative_id: str
    source: str  # "known_sets" | "judged"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def save_triplets(triplets: list[Triplet], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(t.to_json() + "\n" for t in triplets))


def load_triplets(path: str | Path) -> list[Triplet]:
    return [Triplet(**json.loads(line)) for line in Path(path).read_text().splitlines() if line]


class WikipediaSummaries:
    """Fetch lead-section summaries (REST API), cached in memory."""

    def __init__(self, client: httpx.AsyncClient | None = None, concurrency: int = 4) -> None:
        self._client = client
        self._owns = client is None
        self._sem = asyncio.Semaphore(concurrency)
        self._cache: dict[str, str | None] = {}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            from searchprobe.evidence.fetcher import USER_AGENT, _has_h2

            self._client = httpx.AsyncClient(
                http2=_has_h2(),
                timeout=30.0,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and self._owns:
            await self._client.aclose()

    async def summary(self, entity: KnownEntity) -> str | None:
        if not entity.wikipedia_url:
            return None
        if entity.wikipedia_url in self._cache:
            return self._cache[entity.wikipedia_url]
        wiki = wikipedia_title(entity.wikipedia_url)
        if wiki is None:
            return None
        lang, title = wiki
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}"
        text = None
        try:
            async with self._sem:
                resp = await self._http().get(url)
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract") or ""
                text = f"{data.get('title', entity.label)}. {extract}".strip() if extract else None
        except (httpx.HTTPError, ValueError):
            text = None
        self._cache[entity.wikipedia_url] = text
        return text


async def triplets_from_known_sets(
    cases: list[TestCase],
    summaries: WikipediaSummaries,
    per_case: int = 5,
    seed: int = 0,
) -> list[Triplet]:
    """Pair answer-set and negative-set entity summaries for each case."""
    rng = random.Random(seed)
    out: list[Triplet] = []
    for case in cases:
        if not case.answer_set or not case.negative_set:
            continue
        pos = [e for e in case.answer_set if e.wikipedia_url]
        neg = [e for e in case.negative_set if e.wikipedia_url]
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos, neg = pos[: per_case * 2], neg[: per_case * 2]
        pos_text = await asyncio.gather(*[summaries.summary(e) for e in pos])
        neg_text = await asyncio.gather(*[summaries.summary(e) for e in neg])
        pos_pairs = [(e, t) for e, t in zip(pos, pos_text) if t]
        neg_pairs = [(e, t) for e, t in zip(neg, neg_text) if t]
        for (pe, pt), (ne, nt) in list(zip(pos_pairs, neg_pairs))[:per_case]:
            out.append(
                Triplet(
                    case.id,
                    case.category.value,
                    case.text,
                    pt,
                    nt,
                    pe.qid or pe.label,
                    ne.qid or ne.label,
                    "known_sets",
                )
            )
    return out


def triplets_from_judgments(
    cases: dict[str, TestCase],
    judgments: list,
    texts: dict[str, str],
    per_case: int = 5,
    seed: int = 0,
    max_chars: int = 2000,
) -> list[Triplet]:
    """Pair satisfying and violating judged documents of the same case.

    Args:
        texts: content_hash -> document text (from the evidence store).
    """
    from collections import defaultdict

    from searchprobe.evidence.passages import select_passages

    rng = random.Random(seed)
    pos: dict[str, list] = defaultdict(list)
    neg: dict[str, list] = defaultdict(list)
    for j in judgments:
        case = cases.get(j.case_id)
        if case is None or j.content_hash not in texts:
            continue
        if j.is_fully_satisfying(case):
            pos[j.case_id].append(j)
        elif j.is_violating(case):
            neg[j.case_id].append(j)
    out = []
    for cid in sorted(set(pos) & set(neg)):
        case = cases[cid]
        p, n = pos[cid][:], neg[cid][:]
        rng.shuffle(p)
        rng.shuffle(n)
        for pj, nj in list(zip(p, n))[:per_case]:
            out.append(
                Triplet(
                    cid,
                    case.category.value,
                    case.text,
                    select_passages(texts[pj.content_hash], case.text, max_chars=max_chars),
                    select_passages(texts[nj.content_hash], case.text, max_chars=max_chars),
                    pj.content_hash,
                    nj.content_hash,
                    "judged",
                )
            )
    return out
