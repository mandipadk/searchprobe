"""Test cases whose correct answers come from Wikidata.

Each generator runs SPARQL queries to obtain the set of entities that satisfy
the query (``answer_set``) and a set of near-miss entities that break it
(``negative_set``: EU members for "European countries NOT in the EU",
countries just outside a population range, the famous Paris for "Paris,
Texas"). Known attribute values are attached so numeric constraints can be
checked without an LLM. Twins are generated from the same data, so a twin
gets its own correct answer set rather than a guessed one.

Wikidata is incomplete and evolving: ``answer_set_complete`` is only set where
the set is small and well curated, and every case records its SPARQL source.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx

from searchprobe.cases.models import (
    DateConstraint,
    EntityConstraint,
    ExcludeConstraint,
    KnownEntity,
    NumericConstraint,
    Provenance,
    RelationConstraint,
    TestCase,
    TopicConstraint,
)

logger = logging.getLogger(__name__)

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "SearchProbe/3.0 (https://github.com/mandipadk/searchprobe) httpx-sparql"
GENERATOR_VERSION = "1"

PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>
"""

# Countries: UN members plus sovereign states, not dissolved. Wikidata records EU
# membership on Denmark (Q35) and the Netherlands (Q55) rather than on their
# kingdoms (Q756617, Q29999), so the country items are used and the kingdoms
# excluded; otherwise Denmark would appear as a non-member of the EU.
SOVEREIGN = (
    "{ ?item wdt:P463 wd:Q1065 } UNION { ?item wdt:P31 wd:Q3624078 } "
    "UNION { VALUES ?item { wd:Q35 wd:Q55 } } "
    "FILTER(?item NOT IN (wd:Q756617, wd:Q29999)) "
    "FILTER NOT EXISTS { ?item wdt:P576 ?dissolved } "
    # UN membership alone also admits regime items (e.g. "Third Hellenic Republic").
    "FILTER EXISTS { ?item wdt:P31 ?stateClass . FILTER(?stateClass IN (wd:Q6256, wd:Q3624078)) }"
)

CONTINENTS = {
    "Europe": "Q46",
    "Africa": "Q15",
    "Asia": "Q48",
    "South America": "Q18",
    "North America": "Q49",
    "Oceania": "Q55643",
}
EU, NATO, EURO, LANDLOCKED = "Q458", "Q7184", "Q4916", "Q123480"


# Wikidata membership errors found by checking member counts (2026-10). Each entry is
# (country qid, label) and is applied on top of the SPARQL membership pattern.
MEMBERSHIP_ADD: dict[str, list[tuple[str, str]]] = {
    "Q8908": [("Q214", "Slovakia")],  # Council of Europe member since 1993; no statement
    "Q7785": [("Q145", "United Kingdom")],  # Commonwealth: no current statement
}
MEMBERSHIP_REMOVE: dict[str, list[tuple[str, str]]] = {
    "Q7768": [("Q408", "Australia")],  # ASEAN dialogue partner, not a member
    "Q7795": [("Q916", "Angola")],  # left OPEC in January 2024; statement has no end time
}
# Current members that pass SOVEREIGN, used to detect new Wikidata errors at generation.
EXPECTED_MEMBER_COUNTS: dict[str, int] = {
    "Q458": 27,  # EU
    "Q7184": 32,  # NATO
    "Q7768": 11,  # ASEAN (Timor-Leste joined 2025)
    "Q8908": 46,  # Council of Europe
    "Q41550": 38,  # OECD
    "Q19771": 19,  # G20 countries (the EU and African Union are not countries)
    "Q7785": 56,  # Commonwealth of Nations
    "Q7795": 12,  # OPEC
    "Q7172": 22,  # Arab League
    "Q170481": 20,  # APEC's 21 economies minus Hong Kong
}


def member_of(org_qid: str) -> str:
    """Current membership: a P463 statement without an end time (excludes the UK in the EU).

    Wikidata splits some memberships between a country and its kingdom (Denmark's
    EU membership is on Q35, its NATO membership on the Kingdom of Denmark), so
    statements on the kingdom items count for the country items. Known Wikidata errors
    are corrected with MEMBERSHIP_ADD / MEMBERSHIP_REMOVE.
    """
    pattern = (
        "{ { ?item p:P463 ?membership . } UNION { "
        "VALUES (?holder ?item) { (wd:Q756617 wd:Q35) (wd:Q29999 wd:Q55) } "
        "?holder p:P463 ?membership . } "
        f"?membership ps:P463 wd:{org_qid} . "
        "FILTER NOT EXISTS { ?membership pq:P582 ?membershipEnd } }"
    )
    if add := MEMBERSHIP_ADD.get(org_qid):
        pattern += " UNION { VALUES ?item { " + " ".join(f"wd:{q}" for q, _ in add) + " } }"
    if remove := MEMBERSHIP_REMOVE.get(org_qid):
        pattern += " FILTER(?item NOT IN (" + ", ".join(f"wd:{q}" for q, _ in remove) + "))"
    return "{ " + pattern + " }"


class WikidataClient:
    """Minimal SPARQL client with retries and an in-memory cache."""

    def __init__(self, client: httpx.AsyncClient | None = None, max_concurrent: int = 2):
        self._client = client
        self._owns = client is None
        self._cache: dict[str, list[dict[str, str]]] = {}
        self._sem = asyncio.Semaphore(max_concurrent)
        # The search API rate-limits aggressively: one request at a time, spaced out.
        self._search_sem = asyncio.Semaphore(1)

    async def close(self) -> None:
        if self._client is not None and self._owns:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            from searchprobe.evidence.fetcher import _has_h2

            self._client = httpx.AsyncClient(
                http2=_has_h2(),  # Wikimedia rejects many HTTP/1.1 library clients
                timeout=90.0,
                headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
            )
        return self._client

    async def select(self, query: str) -> list[dict[str, str]]:
        """Run a SELECT query; returns rows of {var: value}."""
        full = PREFIXES + query
        if full in self._cache:
            return self._cache[full]
        last: Exception | None = None
        for attempt in range(4):
            try:
                async with self._sem:
                    resp = await self._http().post(ENDPOINT, data={"query": full, "format": "json"})
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Wikidata HTTP {resp.status_code}: {resp.text[:200]}")
                rows = [
                    {k: v["value"] for k, v in b.items()}
                    for b in resp.json()["results"]["bindings"]
                ]
                self._cache[full] = rows
                return rows
            except (httpx.HTTPError, ValueError) as e:
                last = e
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"Wikidata query failed: {last}")

    async def search_entities(self, term: str, limit: int = 150) -> list[str]:
        """QIDs whose label/alias matches ``term`` (Wikidata search API, paginated)."""
        qids: list[str] = []
        offset = 0
        while len(qids) < limit:
            for attempt in range(5):
                async with self._search_sem:
                    await asyncio.sleep(1.0)
                    resp = await self._http().get(
                        "https://www.wikidata.org/w/api.php",
                        params={
                            "action": "wbsearchentities",
                            "search": term,
                            "language": "en",
                            "type": "item",
                            "limit": 50,
                            "continue": offset,
                            "format": "json",
                        },
                    )
                if resp.status_code != 429:
                    break
                retry_after = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                await asyncio.sleep(min(retry_after, 30))
            if resp.status_code >= 400:
                raise RuntimeError(f"Wikidata search HTTP {resp.status_code}")
            data = resp.json()
            qids += [hit["id"] for hit in data.get("search", [])]
            if "search-continue" not in data:
                break
            offset = data["search-continue"]
        return qids[:limit]

    async def entities(
        self, pattern: str, value_expr: str | None = None, value_filter: str | None = None
    ) -> list[KnownEntity]:
        """Entities matching a WHERE-pattern (binding ?item), with optional ?v value.

        ``value_filter`` (e.g. ``?v > 6800``) is applied server-side so large
        classes (all mountains) do not time out.
        """
        value_select = "(MAX(?v) AS ?value)" if value_expr else ""
        value_pattern = (value_expr or "") + (f" FILTER({value_filter})" if value_filter else "")
        query = f"""
        SELECT ?item ?itemLabel ?article {value_select} WHERE {{
          {pattern}
          {value_pattern}
          OPTIONAL {{ ?article schema:about ?item ;
                               schema:isPartOf <https://en.wikipedia.org/> . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} GROUP BY ?item ?itemLabel ?article
        """
        rows = await self.select(query)
        out: dict[str, KnownEntity] = {}
        for r in rows:
            qid = r["item"].rsplit("/", 1)[-1]
            label = r.get("itemLabel", qid)
            if label == qid:  # no English label
                continue
            values = {}
            if "value" in r:
                try:
                    values["value"] = float(r["value"])
                except ValueError:
                    pass
            out[qid] = KnownEntity(
                label=label, qid=qid, wikipedia_url=r.get("article"), values=values
            )
        return sorted(out.values(), key=lambda e: e.label)


def _with_attr(
    entities: list[KnownEntity], attribute: str, scale: float = 1.0
) -> list[KnownEntity]:
    """Rename the generic 'value' to the constraint's attribute name (and rescale)."""
    out = []
    for e in entities:
        values = dict(e.values)
        if "value" in values:
            values[attribute] = values.pop("value") / scale
        out.append(e.model_copy(update={"values": values}))
    return out


def _prov(source: str, notes: str | None = None) -> Provenance:
    return Provenance(
        generator="wikidata",
        version=GENERATOR_VERSION,
        source=" ".join(source.split())[:4000],
        created=datetime.now(UTC).date().isoformat(),
        notes=notes,
    )


def _link(base: TestCase, twin: TestCase, relation: str) -> tuple[TestCase, TestCase]:
    twin = twin.model_copy(update={"twin_of": base.id, "twin_relation": relation})
    base = base.model_copy(update={"twins": {**base.twins, relation: twin.id}})
    return base, twin


@dataclass
class GeneratorSpec:
    name: str
    category: str
    run: Callable[[WikidataClient], Awaitable[list[TestCase]]]


# ---------------------------------------------------------------------------
# Negation and boolean set difference
# ---------------------------------------------------------------------------


def _violating(entities: list[KnownEntity], constraint_id: str) -> list[KnownEntity]:
    """Negatives tagged with the constraint they break, so checkers can decide them."""
    return [e.model_copy(update={"violates": [constraint_id]}) for e in entities]


async def _not_member(
    wd: WikidataClient, continent: str, org_qid: str, org_name: str
) -> list[TestCase]:
    cq = CONTINENTS[continent]
    pattern_all = f"{SOVEREIGN} ?item wdt:P30 wd:{cq} ."
    members = await wd.entities(f"{pattern_all} {member_of(org_qid)}")
    everyone = await wd.entities(pattern_all)
    member_ids = {m.qid for m in members}
    non_members = [e for e in everyone if e.qid not in member_ids]
    if len(members) < 2 or len(non_members) < 2:
        return []
    adjective = {
        "Europe": "European",
        "Africa": "African",
        "Asia": "Asian",
        "South America": "South American",
        "North America": "North American",
        "Oceania": "Oceanian",
    }[continent]
    base = TestCase(
        text=f"{adjective} countries that are not members of {org_name}",
        category="negation",
        operators=["negation"],
        difficulty="medium",
        constraints=[
            TopicConstraint(id="c1", subject=f"a sovereign country in {continent}"),
            ExcludeConstraint(
                id="c2", concept=f"a member state of {org_name}", entity_qids=sorted(member_ids)
            ),
        ],
        answer_set=non_members,
        negative_set=_violating(members, "c2"),
        answer_set_complete=True,
        provenance=_prov(pattern_all + f" minus P463 {org_qid}"),
        validation="auto",
        adversarial_reason="Embeddings of 'not members of X' sit next to 'members of X'",
    )
    twin = TestCase(
        text=f"{adjective} countries that are members of {org_name}",
        category="negation",
        operators=["negation"],
        constraints=[
            TopicConstraint(id="c1", subject=f"a sovereign country in {continent}"),
            TopicConstraint(id="c2", subject=f"a member state of {org_name}"),
        ],
        answer_set=members,
        negative_set=_violating(non_members, "c2"),
        answer_set_complete=True,
        provenance=_prov(pattern_all + " " + member_of(org_qid)),
        validation="auto",
    )
    return list(_link(base, twin, "negation_flip"))


async def _landlocked(wd: WikidataClient, continent: str) -> list[TestCase]:
    cq = CONTINENTS[continent]
    pattern_all = f"{SOVEREIGN} ?item wdt:P30 wd:{cq} ."
    landlocked = await wd.entities(f"{pattern_all} ?item wdt:P31 wd:{LANDLOCKED} .")
    everyone = await wd.entities(pattern_all)
    ids = {e.qid for e in landlocked}
    coastal = [e for e in everyone if e.qid not in ids]
    if len(landlocked) < 2:
        return []
    base = TestCase(
        text=f"Countries in {continent} without a coastline",
        category="negation",
        operators=["negation"],
        constraints=[
            TopicConstraint(id="c1", subject=f"a sovereign country in {continent}"),
            ExcludeConstraint(id="c2", concept="a country with a sea coastline"),
        ],
        answer_set=landlocked,
        negative_set=_violating(coastal, "c2"),
        answer_set_complete=True,
        provenance=_prov(pattern_all + f" ?item wdt:P31 wd:{LANDLOCKED}"),
        validation="auto",
        adversarial_reason="'without a coastline' vs 'with a coastline' differ by one word",
    )
    twin = TestCase(
        text=f"Countries in {continent} with a coastline",
        category="negation",
        operators=["negation"],
        constraints=[
            TopicConstraint(id="c1", subject=f"a sovereign country in {continent}"),
            TopicConstraint(id="c2", subject="a country with a sea coastline"),
        ],
        answer_set=coastal,
        negative_set=_violating(landlocked, "c2"),
        provenance=_prov(pattern_all + f" minus P31 {LANDLOCKED}"),
        validation="auto",
    )
    return list(_link(base, twin, "negation_flip"))


async def _eu_not_euro(wd: WikidataClient) -> list[TestCase]:
    members = await wd.entities(f"{SOVEREIGN} {member_of(EU)}")
    euro = await wd.entities(f"{SOVEREIGN} {member_of(EU)} ?item wdt:P38 wd:{EURO} .")
    euro_ids = {e.qid for e in euro}
    non_euro = [m for m in members if m.qid not in euro_ids]
    if not non_euro:
        return []
    base = TestCase(
        text="European Union member states that do not use the euro",
        category="negation",
        operators=["negation", "conjunction"],
        constraints=[
            TopicConstraint(id="c1", subject="a member state of the European Union"),
            ExcludeConstraint(id="c2", concept="a country whose currency is the euro"),
        ],
        answer_set=non_euro,
        negative_set=_violating(euro, "c2"),
        answer_set_complete=True,
        provenance=_prov(f"EU members (P463 {EU}) minus currency (P38) {EURO}"),
        validation="auto",
    )
    twin = TestCase(
        text="European Union member states that use the euro",
        category="negation",
        operators=["negation", "conjunction"],
        constraints=[
            TopicConstraint(id="c1", subject="a member state of the European Union"),
            TopicConstraint(id="c2", subject="a country whose currency is the euro"),
        ],
        answer_set=euro,
        negative_set=_violating(non_euro, "c2"),
        answer_set_complete=True,
        provenance=_prov(f"EU members with currency (P38) {EURO}"),
        validation="auto",
    )
    return list(_link(base, twin, "negation_flip"))


async def _set_difference(
    wd: WikidataClient, a: tuple[str, str], b: tuple[str, str]
) -> list[TestCase]:
    (a_qid, a_name), (b_qid, b_name) = a, b
    in_a = await wd.entities(f"{SOVEREIGN} {member_of(a_qid)}")
    in_b = await wd.entities(f"{SOVEREIGN} {member_of(b_qid)}")
    b_ids = {e.qid for e in in_b}
    only_a = [e for e in in_a if e.qid not in b_ids]
    both = [e for e in in_a if e.qid in b_ids]
    if len(only_a) < 2 or len(both) < 2:
        return []
    return [
        TestCase(
            text=f"Countries that are members of {a_name} but not of {b_name}",
            category="boolean_logic",
            operators=["boolean", "negation"],
            constraints=[
                TopicConstraint(id="c1", subject=f"a member state of {a_name}"),
                ExcludeConstraint(id="c2", concept=f"a member state of {b_name}"),
            ],
            answer_set=only_a,
            negative_set=_violating(both, "c2"),
            answer_set_complete=True,
            provenance=_prov(f"P463 {a_qid} minus P463 {b_qid}"),
            validation="auto",
            adversarial_reason="A AND NOT B collapses to A AND B in vector space",
        )
    ]


# ---------------------------------------------------------------------------
# Numeric precision
# ---------------------------------------------------------------------------


def _fmt_big(x: float) -> str:
    if x >= 1e6 and x % 1e6 == 0:
        return f"{x / 1e6:g} million"
    return f"{x:,.0f}"


async def _numeric_range(
    wd: WikidataClient,
    *,
    pattern: str,
    value_expr: str,
    attribute: str,
    unit: str,
    noun: str,
    lo: float,
    hi: float,
    scale: float = 1.0,
    text_unit: str = "",
    category: str = "numeric_precision",
) -> list[TestCase]:
    """Entities with lo <= value <= hi; negatives just outside the range (hard negatives)."""
    entities = _with_attr(await wd.entities(pattern, value_expr), attribute, scale)
    inside = [e for e in entities if attribute in e.values and lo <= e.values[attribute] <= hi]
    width = hi - lo
    near = [
        e
        for e in entities
        if attribute in e.values
        and (lo - width <= e.values[attribute] < lo or hi < e.values[attribute] <= hi + width)
    ]
    if len(inside) < 2:
        return []

    def make(lo_: float, hi_: float, ans, neg) -> TestCase:
        return TestCase(
            text=f"{noun} with a {attribute} between {_fmt_big(lo_)} and {_fmt_big(hi_)}"
            f"{text_unit}",
            category=category,
            operators=["numeric"],
            difficulty="hard",
            constraints=[
                TopicConstraint(id="c1", subject=noun.lower()),
                NumericConstraint(
                    id="c2", attribute=attribute, op="between", value=lo_, value_max=hi_, unit=unit
                ),
            ],
            answer_set=ans,
            negative_set=_violating(neg, "c2"),
            provenance=_prov(pattern + " " + value_expr),
            validation="auto",
            adversarial_reason="Numbers are tokens to an embedding model, not magnitudes",
        )

    base = make(lo, hi, inside, near)
    # Twin: the next decade up; correct answers come from the same data.
    lo2, hi2 = lo * 10, hi * 10
    inside2 = [e for e in entities if attribute in e.values and lo2 <= e.values[attribute] <= hi2]
    if len(inside2) < 2:
        return [base]
    twin = make(lo2, hi2, inside2, inside)
    return list(_link(base, twin, "numeric_x10"))


async def _numeric_threshold(
    wd: WikidataClient,
    *,
    pattern: str,
    value_expr: str,
    attribute: str,
    unit: str,
    noun: str,
    threshold: float,
    complete: bool = False,
    text_unit: str = "",
    scale: float = 1.0,
) -> list[TestCase]:
    entities = _with_attr(
        await wd.entities(pattern, value_expr, value_filter=f"?v > {threshold * scale * 0.85}"),
        attribute,
        scale,
    )
    above = [e for e in entities if e.values.get(attribute, -1) > threshold]
    near = [e for e in entities if threshold * 0.85 <= e.values.get(attribute, -1) <= threshold]
    if len(above) < 2:
        return []
    return [
        TestCase(
            text=f"{noun} with a {attribute} of more than {threshold:,.0f}{text_unit}",
            category="numeric_precision",
            operators=["numeric"],
            difficulty="hard",
            constraints=[
                TopicConstraint(id="c1", subject=noun.lower()),
                NumericConstraint(
                    id="c2", attribute=attribute, op="gt", value=threshold, unit=unit
                ),
            ],
            answer_set=above,
            negative_set=_violating(near, "c2"),
            answer_set_complete=complete,
            provenance=_prov(pattern + " " + value_expr),
            validation="auto",
        )
    ]


def _best_quantity(pid: str) -> str:
    """Normalized quantity from best-ranked statements only (preferred, else normal):
    deprecated values such as a superseded height never become answers."""
    return (
        f"?item p:{pid} ?st . ?st a wikibase:BestRank ; psn:{pid} ?n . "
        "?n wikibase:quantityAmount ?v ."
    )


POPULATION = "?item wdt:P1082 ?v ."
AREA_KM2 = _best_quantity("P2046")
HEIGHT_M = _best_quantity("P2048")
ELEVATION_M = _best_quantity("P2044")
LENGTH_M = _best_quantity("P2043")


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

NOBEL = {
    "Physics": "Q38104",
    "Chemistry": "Q44585",
    "Physiology or Medicine": "Q80061",
    "Literature": "Q37922",
    "Peace": "Q35637",
}


async def _nobel_year(wd: WikidataClient, field: str, year: int) -> list[TestCase]:
    qid = NOBEL[field]

    async def winners(y: int) -> list[KnownEntity]:
        return await wd.entities(
            f"?item p:P166 ?aw . ?aw ps:P166 wd:{qid} ; pq:P585 ?d . FILTER(YEAR(?d) = {y})"
        )

    this_year, prev_year = await winners(year), await winners(year - 1)
    if not this_year:
        return []

    def make(y: int, ans, neg) -> TestCase:
        return TestCase(
            text=f"Winners of the Nobel Prize in {field} in {y}",
            category="temporal_constraint",
            operators=["temporal"],
            constraints=[
                TopicConstraint(id="c1", subject=f"a laureate of the Nobel Prize in {field}"),
                DateConstraint(
                    id="c2",
                    field="event",
                    start=date(y, 1, 1),
                    end=date(y, 12, 31),
                    description=f"The prize was awarded in {y}",
                ),
            ],
            answer_set=ans,
            negative_set=_violating(neg, "c2"),
            answer_set_complete=True,
            provenance=_prov(f"P166 {qid} with P585 year {y}"),
            validation="auto",
            adversarial_reason="Adjacent years are near-identical in embedding space",
        )

    base = make(year, this_year, prev_year)
    if not prev_year:
        return [base]
    twin = make(year - 1, prev_year, this_year)
    return list(_link(base, twin, "date_shift_-1y"))


async def _independence_year(wd: WikidataClient, year: int) -> list[TestCase]:
    async def founded(y: int) -> list[KnownEntity]:
        return await wd.entities(f"{SOVEREIGN} ?item wdt:P571 ?d . FILTER(YEAR(?d) = {y})")

    ans, neg = await founded(year), await founded(year - 1)
    if len(ans) < 2:
        return []
    return [
        TestCase(
            text=f"Countries that became independent in {year}",
            category="temporal_constraint",
            operators=["temporal"],
            constraints=[
                TopicConstraint(id="c1", subject="a sovereign country"),
                DateConstraint(
                    id="c2",
                    field="event",
                    start=date(year, 1, 1),
                    end=date(year, 12, 31),
                    description=f"The country became independent in {year}",
                ),
            ],
            answer_set=ans,
            negative_set=_violating(neg, "c2"),
            provenance=_prov(f"sovereign state with P571 year {year}"),
            validation="auto",
        )
    ]


# ---------------------------------------------------------------------------
# Multi-constraint, entity disambiguation, compositional
# ---------------------------------------------------------------------------


async def _landlocked_population(
    wd: WikidataClient, continent: str, min_pop: float
) -> list[TestCase]:
    cq = CONTINENTS[continent]
    pattern = f"{SOVEREIGN} ?item wdt:P30 wd:{cq} . ?item wdt:P31 wd:{LANDLOCKED} ."
    entities = _with_attr(await wd.entities(pattern, POPULATION), "population")
    ans = [e for e in entities if e.values.get("population", 0) > min_pop]
    small = [e for e in entities if 0 < e.values.get("population", 0) <= min_pop]
    coastal_big = _with_attr(
        await wd.entities(f"{SOVEREIGN} ?item wdt:P30 wd:{cq} .", POPULATION), "population"
    )
    ids = {e.qid for e in entities}
    coastal_big = [
        e for e in coastal_big if e.qid not in ids and e.values.get("population", 0) > min_pop
    ]
    if len(ans) < 2:
        return []
    for e in coastal_big:
        e.violates = ["c2"]
    for e in small:
        e.violates = ["c3"]
    return [
        TestCase(
            text=f"Landlocked countries in {continent} with a population of more than "
            f"{_fmt_big(min_pop)}",
            category="multi_constraint",
            operators=["conjunction", "numeric"],
            difficulty="hard",
            constraints=[
                TopicConstraint(id="c1", subject=f"a sovereign country in {continent}"),
                TopicConstraint(id="c2", subject="a landlocked country (no sea coastline)"),
                NumericConstraint(id="c3", attribute="population", op="gt", value=min_pop),
            ],
            answer_set=ans,
            negative_set=small + coastal_big,
            answer_set_complete=True,
            provenance=_prov(pattern + " " + POPULATION),
            validation="auto",
            adversarial_reason="Each result must satisfy all three constraints, not most",
        )
    ]


async def _high_capitals(wd: WikidataClient, continent: str, min_elev: float) -> list[TestCase]:
    cq = CONTINENTS[continent]
    pattern = f"{SOVEREIGN.replace('?item', '?country')} ?country wdt:P30 wd:{cq} ; wdt:P36 ?item ."
    entities = _with_attr(await wd.entities(pattern, ELEVATION_M), "elevation")
    ans = [e for e in entities if e.values.get("elevation", -1) > min_elev]
    low = [e for e in entities if 0 <= e.values.get("elevation", -1) <= min_elev]
    if len(ans) < 2:
        return []
    for e in low:
        e.violates = ["c2"]
    return [
        TestCase(
            text=f"Capital cities in {continent} located more than {min_elev:,.0f} meters "
            "above sea level",
            category="multi_constraint",
            operators=["conjunction", "numeric"],
            difficulty="hard",
            constraints=[
                TopicConstraint(id="c1", subject=f"the capital city of a country in {continent}"),
                NumericConstraint(
                    id="c2", attribute="elevation", op="gt", value=min_elev, unit="m"
                ),
            ],
            answer_set=ans,
            negative_set=low,
            answer_set_complete=True,
            provenance=_prov(pattern + " " + ELEVATION_M),
            validation="auto",
        )
    ]


FAMOUS_CITIES = {  # label -> (famous QID, country)
    "Paris": ("Q90", "France"),
    "Athens": ("Q1524", "Greece"),
    "Rome": ("Q220", "Italy"),
    "Moscow": ("Q649", "Russia"),
    "Berlin": ("Q64", "Germany"),
    "Dublin": ("Q1761", "Ireland"),
    "Cairo": ("Q85", "Egypt"),
    "Lisbon": ("Q597", "Portugal"),
    "Toledo": ("Q5836", "Spain"),
    "Alexandria": ("Q87", "Egypt"),
    "Florence": ("Q2044", "Italy"),
    "Naples": ("Q2634", "Italy"),
    "Manchester": ("Q18125", "England"),
    "Birmingham": ("Q2256", "England"),
    "Cambridge": ("Q350", "England"),
    "Oxford": ("Q34217", "England"),
    "Vienna": ("Q1741", "Austria"),
    "Warsaw": ("Q270", "Poland"),
    "Delhi": ("Q1353", "India"),
    "Venice": ("Q641", "Italy"),
    "Lima": ("Q2868", "Peru"),
    "Troy": ("Q22647", "Turkey"),
    "Sparta": ("Q5690", "Greece"),
    "Bethlehem": ("Q5776", "Palestine"),
    "Amsterdam": ("Q727", "the Netherlands"),
    "Hamburg": ("Q1055", "Germany"),
    "Geneva": ("Q71", "Switzerland"),
    "Milan": ("Q490", "Italy"),
    "Versailles": ("Q621", "France"),
}


async def _namesake_cities(wd: WikidataClient, label: str) -> list[TestCase]:
    famous_qid, famous_country = FAMOUS_CITIES[label]
    # Entity search runs server-side (WDQS mwapi), avoiding the rate-limited search API.
    rows = await wd.select(
        f"""
        SELECT ?item ?stateLabel ?article (MAX(?pop) AS ?population) WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org" ;
                            wikibase:api "EntitySearch" ;
                            mwapi:search "{label}" ;
                            mwapi:language "en" ;
                            mwapi:limit "max" .
            ?item wikibase:apiOutputItem mwapi:item .
          }}
          ?item rdfs:label "{label}"@en ;                 # exact label, not an alias
                wdt:P31/wdt:P279* wd:Q486972 ;              # a human settlement
                wdt:P17 wd:Q30 ; wdt:P131+ ?state ; wdt:P1082 ?pop .
          ?state wdt:P31 wd:Q35657 .
          OPTIONAL {{ ?article schema:about ?item ;
                               schema:isPartOf <https://en.wikipedia.org/> . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} GROUP BY ?item ?stateLabel ?article ORDER BY DESC(?population) LIMIT 1
        """
    )
    if not rows or float(rows[0].get("population", 0)) < 5000:
        return []
    row = rows[0]
    qid = row["item"].rsplit("/", 1)[-1]
    state = row["stateLabel"]
    famous = await wd.entities(f"VALUES ?item {{ wd:{famous_qid} }}")
    us_city = KnownEntity(
        label=f"{label}, {state}",
        qid=qid,
        wikipedia_url=row.get("article"),
        aliases=[f"{label} {state}"],
    )
    return [
        TestCase(
            text=f"History and economy of {label}, {state}",
            category="entity_disambiguation",
            operators=["disambiguation"],
            constraints=[
                EntityConstraint(
                    id="c1",
                    name=f"{label}, {state} (United States)",
                    qid=qid,
                    wrong=[f"{label}, {famous_country}"],
                ),
            ],
            answer_set=[us_city],
            negative_set=[e.model_copy(update={"violates": ["c1"]}) for e in famous],
            answer_set_complete=True,
            provenance=_prov(f'US place labelled "{label}" (largest), vs {famous_qid}'),
            validation="auto",
            adversarial_reason=f"{label}, {famous_country} dominates the name in embedding space",
        )
    ]


RIVERS = {
    "Danube": "Q1653",
    "Rhine": "Q584",
    "Mississippi River": "Q1497",
    "Amazon River": "Q3783",
    "Elbe": "Q1644",
    "Volga": "Q626",
    "Ganges": "Q5089",
    "Nile": "Q3392",
    "Rhône": "Q602",
    "Mekong River": "Q41179",
    "Yangtze River": "Q5413",
    "Dnieper": "Q40855",
    "Po": "Q643",
    "Seine": "Q1471",
    "River Thames": "Q19686",
    "Missouri River": "Q5419",
    "Ohio River": "Q4915",
}


async def _tributaries(wd: WikidataClient, river: str) -> list[TestCase]:
    rq = RIVERS[river]
    tributaries = await wd.entities(f"?item wdt:P403 wd:{rq} .")
    mouth = await wd.entities(f"wd:{rq} wdt:P403 ?item .")
    if len(tributaries) < 3:
        return []
    name = river if river.endswith("River") else f"the {river}"
    for m in mouth:
        m.violates = ["c1"]
    return [
        TestCase(
            text=f"Rivers that flow into {name}",
            category="compositional",
            operators=["relation"],
            constraints=[
                RelationConstraint(
                    id="c1", subject="the river in the result", predicate="flows into", object=name
                ),
            ],
            answer_set=tributaries,
            negative_set=mouth,
            provenance=_prov(f"?item wdt:P403 wd:{rq}"),
            validation="auto",
            adversarial_reason=f"'flows into {name}' vs '{name} flows into' share every word",
        )
    ]


# Every hard-coded QID with the English label it must have. Checked against
# Wikidata before generation so a wrong id fails loudly instead of silently
# producing a case with the wrong answer set (this caught "Q170481", which is
# APEC, not Mercosur).
ORGS: dict[str, tuple[str, str]] = {  # qid -> (expected label substring, display name)
    EU: ("European Union", "the European Union"),
    NATO: ("NATO", "NATO"),
    "Q7768": ("ASEAN", "ASEAN"),
    "Q8908": ("Council of Europe", "the Council of Europe"),
    "Q41550": ("Organization for Economic Cooperation and Development", "the OECD"),
    "Q19771": ("G20", "the G20"),
    "Q7785": ("Commonwealth of Nations", "the Commonwealth of Nations"),
    "Q7795": ("Organization of the Petroleum Exporting Countries", "OPEC"),
    "Q7172": ("Arab League", "the Arab League"),
    "Q170481": ("Asia-Pacific Economic Cooperation", "APEC"),
}
EXPECTED_LABELS: dict[str, str] = {
    **{q: v[0] for q, v in ORGS.items()},
    # EURO (Q4916) is omitted: WDQS returns no English label for it, but it is the
    # currency (P38) of Germany, France, etc., which is what the generator relies on.
    LANDLOCKED: "landlocked country",
    "Q8502": "mountain",
    "Q11303": "skyscraper",
    "Q4022": "river",
    "Q23397": "lake",
    **{q: name for name, q in CONTINENTS.items()},
    **{
        q: ("Nobel Peace Prize" if f == "Peace" else f"Nobel Prize in {f}")
        for f, q in NOBEL.items()
    },
    **{q: name.replace(" River", "") for name, q in RIVERS.items()},
    **{q: label for label, (q, _) in FAMOUS_CITIES.items()},
    **{
        q: label
        for fixes in (MEMBERSHIP_ADD, MEMBERSHIP_REMOVE)
        for items in fixes.values()
        for q, label in items
    },
}


async def verify_constants(wd: WikidataClient, expected: dict[str, str] | None = None) -> list[str]:
    """Return mismatches between hard-coded QIDs and their English labels."""
    expected = expected or EXPECTED_LABELS
    values = " ".join(f"wd:{q}" for q in expected)
    rows = await wd.select(
        f"""SELECT ?item ?itemLabel WHERE {{ VALUES ?item {{ {values} }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}"""
    )
    labels = {r["item"].rsplit("/", 1)[-1]: r.get("itemLabel", "") for r in rows}
    problems = []
    for qid, want in expected.items():
        got = labels.get(qid, "")
        if want.lower() not in got.lower():
            problems.append(f"{qid}: expected label containing {want!r}, Wikidata has {got!r}")
    return problems


async def verify_memberships(
    wd: WikidataClient, expected: dict[str, int] | None = None
) -> list[str]:
    """Return organizations whose (corrected) current member count is not the expected one."""
    expected = expected or EXPECTED_MEMBER_COUNTS
    problems = []
    for org, want in expected.items():
        rows = await wd.select(
            f"SELECT (COUNT(DISTINCT ?item) AS ?n) WHERE {{ {SOVEREIGN} {member_of(org)} }}"
        )
        got = int(rows[0]["n"]) if rows else 0
        if got != want:
            problems.append(f"{org}: expected {want} current members, Wikidata gives {got}")
    return problems


def default_specs() -> list[GeneratorSpec]:
    """The generator catalogue used to build the shipped dataset."""
    specs: list[GeneratorSpec] = []

    def add(name: str, category: str, fn: Callable[[WikidataClient], Awaitable[list[TestCase]]]):
        specs.append(GeneratorSpec(name, category, fn))

    # Negation: "<continent> countries that are not members of <org>" (+ twin).
    for cont, org in [
        ("Europe", EU),
        ("Europe", NATO),
        ("Europe", "Q8908"),
        ("Europe", "Q41550"),
        ("Asia", "Q7768"),
        ("Asia", "Q7172"),
        ("Asia", "Q41550"),
        ("Asia", "Q7785"),
        ("Asia", "Q7795"),
        ("Africa", "Q7785"),
        ("Africa", "Q7172"),
        ("Africa", "Q7795"),
        ("South America", "Q41550"),
        ("North America", "Q41550"),
        ("Oceania", "Q7785"),
        ("Asia", "Q170481"),
    ]:
        add(
            f"not_member:{cont}:{org}",
            "negation",
            lambda wd, c=cont, o=org: _not_member(wd, c, o, ORGS[o][1]),
        )
    for cont in ("Africa", "Asia", "Europe"):  # South America has only two
        add(f"landlocked:{cont}", "negation", lambda wd, c=cont: _landlocked(wd, c))
    add("eu_not_euro", "negation", _eu_not_euro)

    # Boolean: set differences between organizations.
    for a, b in [
        (NATO, EU),
        (EU, NATO),
        ("Q41550", EU),
        (EU, "Q41550"),
        ("Q19771", "Q41550"),
        ("Q41550", "Q19771"),
        ("Q8908", EU),
        ("Q7795", "Q7172"),
        ("Q7172", "Q7795"),
        ("Q7785", "Q19771"),
        ("Q170481", "Q41550"),
    ]:
        add(
            f"difference:{a}:{b}",
            "boolean_logic",
            lambda wd, a=a, b=b: _set_difference(wd, (a, ORGS[a][1]), (b, ORGS[b][1])),
        )

    # Numeric ranges (x10 twins chosen so no twin duplicates another base case).
    for lo, hi in [(1e6, 2e6), (3e6, 4e6), (5e6, 6e6), (7e6, 8e6), (2e6, 2.5e6)]:
        add(
            f"population:{lo:g}-{hi:g}",
            "numeric_precision",
            lambda wd, lo=lo, hi=hi: _numeric_range(
                wd,
                pattern=SOVEREIGN,
                value_expr=POPULATION,
                attribute="population",
                unit="people",
                noun="Countries",
                lo=lo,
                hi=hi,
            ),
        )
    for lo, hi in [(10_000, 20_000), (30_000, 40_000), (50_000, 60_000), (2_000, 5_000)]:
        add(
            f"area:{lo}-{hi}",
            "numeric_precision",
            lambda wd, lo=lo, hi=hi: _numeric_range(
                wd,
                pattern=SOVEREIGN,
                value_expr=AREA_KM2,
                attribute="area",
                unit="km²",
                noun="Countries",
                lo=lo,
                hi=hi,
                scale=1e6,
                text_unit=" square kilometers",
            ),
        )
    thresholds = [
        (
            "eight_thousanders",
            "?item wdt:P31 wd:Q8502 .",
            ELEVATION_M,
            "elevation",
            "m",
            "Mountains",
            8000,
            True,
            " meters",
            1.0,
        ),
        (
            "skyscrapers_500m",
            "?item wdt:P31 wd:Q11303 .",
            HEIGHT_M,
            "height",
            "m",
            "Skyscrapers",
            500,
            False,
            " meters",
            1.0,
        ),
        (
            "skyscrapers_400m",
            "?item wdt:P31 wd:Q11303 .",
            HEIGHT_M,
            "height",
            "m",
            "Skyscrapers",
            400,
            False,
            " meters",
            1.0,
        ),
        (
            "rivers_4000km",
            "?item wdt:P31 wd:Q4022 .",
            LENGTH_M,
            "length",
            "km",
            "Rivers",
            4000,
            False,
            " kilometers",
            1000.0,
        ),
        (
            "lakes_20000km2",
            "?item wdt:P31 wd:Q23397 .",
            AREA_KM2,
            "area",
            "km²",
            "Lakes",
            20000,
            False,
            " square kilometers",
            1e6,
        ),
    ]
    for name, pattern, expr, attr, unit, noun, thr, complete, text_unit, scale in thresholds:
        add(
            name,
            "numeric_precision",
            lambda wd, p=pattern, e=expr, a=attr, u=unit, n=noun, t=thr, c=complete, tu=text_unit, sc=scale: (
                _numeric_threshold(
                    wd,
                    pattern=p,
                    value_expr=e,
                    attribute=a,
                    unit=u,
                    noun=n,
                    threshold=t,
                    complete=c,
                    text_unit=tu,
                    scale=sc,
                )
            ),
        )

    # Temporal: Nobel laureates by year (twins are the previous year; years are
    # spaced so no twin duplicates a base case) and independence years.
    for field in NOBEL:
        for year in (2015, 2017, 2019, 2021, 2023):
            add(
                f"nobel:{field}:{year}",
                "temporal_constraint",
                lambda wd, f=field, y=year: _nobel_year(wd, f, y),
            )
    for year in (1960, 1962, 1966, 1971, 1975, 1991, 1993):
        add(
            f"independence:{year}",
            "temporal_constraint",
            lambda wd, y=year: _independence_year(wd, y),
        )

    # Multi-constraint.
    for cont, pop in [
        ("Europe", 5e6),
        ("Europe", 1e7),
        ("Africa", 2e7),
        ("Africa", 1e7),
        ("Asia", 1e7),
    ]:
        add(
            f"landlocked_pop:{cont}:{pop:g}",
            "multi_constraint",
            lambda wd, c=cont, p=pop: _landlocked_population(wd, c, p),
        )
    for cont, elev in [
        ("South America", 2000),
        ("Africa", 1000),
        ("Asia", 1000),
        ("Europe", 500),
        ("North America", 1000),
    ]:
        add(
            f"high_capitals:{cont}",
            "multi_constraint",
            lambda wd, c=cont, e=elev: _high_capitals(wd, c, e),
        )

    # Entity disambiguation and compositional.
    for label in FAMOUS_CITIES:
        add(
            f"namesake:{label}",
            "entity_disambiguation",
            lambda wd, lb=label: _namesake_cities(wd, lb),
        )
    for river in RIVERS:
        add(f"tributaries:{river}", "compositional", lambda wd, r=river: _tributaries(wd, r))
    return specs


async def generate(
    specs: list[GeneratorSpec] | None = None,
    client: WikidataClient | None = None,
    on_progress: Callable[[str, int], None] | None = None,
) -> tuple[list[TestCase], list[str]]:
    """Run generator specs; returns (cases, failure messages)."""
    specs = default_specs() if specs is None else specs
    wd = client or WikidataClient()
    cases: list[TestCase] = []
    failures: list[str] = []

    async def _run(spec: GeneratorSpec) -> None:
        try:
            produced = await spec.run(wd)
        except Exception as e:  # keep going: one bad query must not sink the dataset
            failures.append(f"{spec.name}: {e}")
            logger.warning("Generator %s failed: %s", spec.name, e)
            return
        if not produced:
            failures.append(f"{spec.name}: produced no case (too few entities)")
        cases.extend(produced)
        if on_progress:
            on_progress(spec.name, len(produced))

    try:
        problems = await verify_constants(wd)
        if problems:
            raise RuntimeError(
                "Hard-coded Wikidata ids do not match their labels:\n  " + "\n  ".join(problems)
            )
        problems = await verify_memberships(wd)
        if problems:
            raise RuntimeError(
                "Membership data disagrees with known member counts (update MEMBERSHIP_ADD/"
                "MEMBERSHIP_REMOVE or EXPECTED_MEMBER_COUNTS):\n  " + "\n  ".join(problems)
            )
        await asyncio.gather(*[_run(spec) for spec in specs])
    finally:
        if client is None:
            await wd.close()
    cases.sort(key=lambda c: (c.category.value, c.text))
    return cases, failures
