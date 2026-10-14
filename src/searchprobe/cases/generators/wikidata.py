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

# Current sovereign states (excludes dissolved ones).
SOVEREIGN = "?item wdt:P31 wd:Q3624078 . FILTER NOT EXISTS { ?item wdt:P576 ?dissolved }"

CONTINENTS = {
    "Europe": "Q46",
    "Africa": "Q15",
    "Asia": "Q48",
    "South America": "Q18",
    "North America": "Q49",
    "Oceania": "Q55643",
}
EU, NATO, EURO, LANDLOCKED = "Q458", "Q7184", "Q4916", "Q123480"


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


async def _not_member(
    wd: WikidataClient, continent: str, org_qid: str, org_name: str
) -> list[TestCase]:
    cq = CONTINENTS[continent]
    pattern_all = f"{SOVEREIGN} ?item wdt:P30 wd:{cq} ."
    members = await wd.entities(f"{pattern_all} ?item wdt:P463 wd:{org_qid} .")
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
        negative_set=members,
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
        negative_set=non_members,
        answer_set_complete=True,
        provenance=_prov(pattern_all + f" ?item wdt:P463 wd:{org_qid}"),
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
        negative_set=coastal,
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
        negative_set=landlocked,
        provenance=_prov(pattern_all + f" minus P31 {LANDLOCKED}"),
        validation="auto",
    )
    return list(_link(base, twin, "negation_flip"))


async def _eu_not_euro(wd: WikidataClient) -> list[TestCase]:
    members = await wd.entities(f"{SOVEREIGN} ?item wdt:P463 wd:{EU} .")
    euro = await wd.entities(f"{SOVEREIGN} ?item wdt:P463 wd:{EU} . ?item wdt:P38 wd:{EURO} .")
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
        negative_set=euro,
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
        negative_set=non_euro,
        answer_set_complete=True,
        provenance=_prov(f"EU members with currency (P38) {EURO}"),
        validation="auto",
    )
    return list(_link(base, twin, "negation_flip"))


async def _set_difference(
    wd: WikidataClient, a: tuple[str, str], b: tuple[str, str]
) -> list[TestCase]:
    (a_qid, a_name), (b_qid, b_name) = a, b
    in_a = await wd.entities(f"{SOVEREIGN} ?item wdt:P463 wd:{a_qid} .")
    in_b = await wd.entities(f"{SOVEREIGN} ?item wdt:P463 wd:{b_qid} .")
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
            negative_set=both,
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
            negative_set=neg,
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
) -> list[TestCase]:
    entities = _with_attr(
        await wd.entities(pattern, value_expr, value_filter=f"?v > {threshold * 0.85}"), attribute
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
            negative_set=near,
            answer_set_complete=complete,
            provenance=_prov(pattern + " " + value_expr),
            validation="auto",
        )
    ]


POPULATION = "?item wdt:P1082 ?v ."
AREA_KM2 = "?item p:P2046 ?st . ?st psn:P2046 ?n . ?n wikibase:quantityAmount ?v ."
HEIGHT_M = "?item p:P2048 ?st . ?st psn:P2048 ?n . ?n wikibase:quantityAmount ?v ."
ELEVATION_M = "?item p:P2044 ?st . ?st psn:P2044 ?n . ?n wikibase:quantityAmount ?v ."
LENGTH_M = "?item p:P2043 ?st . ?st psn:P2043 ?n . ?n wikibase:quantityAmount ?v ."


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
            negative_set=neg,
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
            negative_set=neg,
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
    pattern = f"?country wdt:P31 wd:Q3624078 ; wdt:P30 wd:{cq} ; wdt:P36 ?item ."
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
    "Sydney": ("Q3130", "Australia"),
    "Vienna": ("Q1741", "Austria"),
    "Warsaw": ("Q270", "Poland"),
    "Delhi": ("Q1353", "India"),
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
    "Elbe": "Q1676",
    "Volga": "Q626",
    "Ganges": "Q5089",
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


def default_specs() -> list[GeneratorSpec]:
    """The generator catalogue used to build the shipped dataset."""
    specs: list[GeneratorSpec] = []
    for cont, org, name in [
        ("Europe", EU, "the European Union"),
        ("Europe", NATO, "NATO"),
        ("Africa", "Q7159", "the African Union"),
        ("Asia", "Q7768", "ASEAN"),
        ("South America", "Q170481", "Mercosur"),
    ]:
        specs.append(
            GeneratorSpec(
                f"not_member:{cont}:{org}",
                "negation",
                lambda wd, c=cont, o=org, n=name: _not_member(wd, c, o, n),
            )
        )
    for cont in ("Africa", "Asia", "Europe", "South America"):
        specs.append(
            GeneratorSpec(f"landlocked:{cont}", "negation", lambda wd, c=cont: _landlocked(wd, c))
        )
    specs.append(GeneratorSpec("eu_not_euro", "negation", _eu_not_euro))
    for a, b in [
        ((NATO, "NATO"), (EU, "the European Union")),
        ((EU, "the European Union"), (NATO, "NATO")),
        (
            ("Q8908", "the Commonwealth of Nations"),
            ("Q1065", "the United Nations Security Council"),
        ),
    ]:
        specs.append(
            GeneratorSpec(
                f"difference:{a[0]}:{b[0]}",
                "boolean_logic",
                lambda wd, a=a, b=b: _set_difference(wd, a, b),
            )
        )

    for lo, hi in [(1e6, 2e6), (5e6, 6e6), (2e7, 3e7)]:
        specs.append(
            GeneratorSpec(
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
        )
    for lo, hi in [(10_000, 20_000), (100_000, 120_000)]:
        specs.append(
            GeneratorSpec(
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
        )
    specs.append(
        GeneratorSpec(
            "eight_thousanders",
            "numeric_precision",
            lambda wd: _numeric_threshold(
                wd,
                pattern="?item wdt:P31 wd:Q8502 .",
                value_expr=ELEVATION_M,
                attribute="elevation",
                unit="m",
                noun="Mountains",
                threshold=8000,
                complete=True,
                text_unit=" meters",
            ),
        )
    )
    specs.append(
        GeneratorSpec(
            "skyscrapers_500m",
            "numeric_precision",
            lambda wd: _numeric_threshold(
                wd,
                pattern="?item wdt:P31 wd:Q11303 .",
                value_expr=HEIGHT_M,
                attribute="height",
                unit="m",
                noun="Skyscrapers",
                threshold=500,
                text_unit=" meters",
            ),
        )
    )
    specs.append(
        GeneratorSpec(
            "rivers_4000km",
            "numeric_precision",
            lambda wd: _numeric_threshold(
                wd,
                pattern="?item wdt:P31 wd:Q4022 .",
                value_expr=LENGTH_M,
                attribute="length",
                unit="m",
                noun="Rivers",
                threshold=4_000_000,
                text_unit=" meters",
            ),
        )
    )

    for field in NOBEL:
        for year in (2018, 2021, 2023):
            specs.append(
                GeneratorSpec(
                    f"nobel:{field}:{year}",
                    "temporal_constraint",
                    lambda wd, f=field, y=year: _nobel_year(wd, f, y),
                )
            )
    for year in (1960, 1991, 1993):
        specs.append(
            GeneratorSpec(
                f"independence:{year}",
                "temporal_constraint",
                lambda wd, y=year: _independence_year(wd, y),
            )
        )

    for cont, pop in [("Europe", 5e6), ("Africa", 2e7), ("Asia", 1e7), ("South America", 5e6)]:
        specs.append(
            GeneratorSpec(
                f"landlocked_pop:{cont}",
                "multi_constraint",
                lambda wd, c=cont, p=pop: _landlocked_population(wd, c, p),
            )
        )
    for cont, elev in [("South America", 2000), ("Africa", 1000), ("Asia", 1000)]:
        specs.append(
            GeneratorSpec(
                f"high_capitals:{cont}",
                "multi_constraint",
                lambda wd, c=cont, e=elev: _high_capitals(wd, c, e),
            )
        )
    for label in FAMOUS_CITIES:
        specs.append(
            GeneratorSpec(
                f"namesake:{label}",
                "entity_disambiguation",
                lambda wd, lb=label: _namesake_cities(wd, lb),
            )
        )
    for river in RIVERS:
        specs.append(
            GeneratorSpec(
                f"tributaries:{river}", "compositional", lambda wd, r=river: _tributaries(wd, r)
            )
        )
    return specs


async def generate(
    specs: list[GeneratorSpec] | None = None,
    client: WikidataClient | None = None,
    on_progress: Callable[[str, int], None] | None = None,
) -> tuple[list[TestCase], list[str]]:
    """Run generator specs; returns (cases, failure messages)."""
    specs = specs or default_specs()
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
        await asyncio.gather(*[_run(spec) for spec in specs])
    finally:
        if client is None:
            await wd.close()
    cases.sort(key=lambda c: (c.category.value, c.text))
    return cases, failures
