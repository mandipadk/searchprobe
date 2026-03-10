"""Tests for the research session DAG."""

import pytest
from dataclasses import dataclass, field

from searchprobe.core.protocols import AnalysisResult
from searchprobe.core.signals import Signal, SignalType
from searchprobe.intelligence.session import ResearchSession, SharedContext


@dataclass
class MockStage:
    name: str
    depends_on: list[str] = field(default_factory=list)
    executed: bool = False

    async def execute(self, context: SharedContext) -> AnalysisResult:
        self.executed = True
        return AnalysisResult(
            analysis_type=self.name,
            categories=["test"],
            summary={"stage": self.name},
        )


class TestTopologicalSort:
    def test_no_dependencies(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="a"))
        session.add_stage(MockStage(name="b"))
        order = session._topological_sort()
        assert set(order) == {"a", "b"}

    def test_linear_chain(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="a"))
        session.add_stage(MockStage(name="b", depends_on=["a"]))
        session.add_stage(MockStage(name="c", depends_on=["b"]))
        order = session._topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="a"))
        session.add_stage(MockStage(name="b", depends_on=["a"]))
        session.add_stage(MockStage(name="c", depends_on=["a"]))
        session.add_stage(MockStage(name="d", depends_on=["b", "c"]))
        order = session._topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_circular_raises(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="a", depends_on=["b"]))
        session.add_stage(MockStage(name="b", depends_on=["a"]))
        with pytest.raises(ValueError, match="Circular"):
            session._topological_sort()


class TestSessionRun:
    @pytest.mark.asyncio
    async def test_executes_all_stages(self):
        session = ResearchSession()
        stages = [MockStage(name="a"), MockStage(name="b", depends_on=["a"])]
        for s in stages:
            session.add_stage(s)

        context = await session.run()
        assert all(s.executed for s in stages)
        assert "a" in context.results
        assert "b" in context.results

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="only"))

        calls = []
        await session.run(progress_callback=lambda name, c, t: calls.append((name, c, t)))
        assert len(calls) == 2  # "only" + "done"


class TestTopologicalLevels:
    def test_independent_stages_single_level(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="a"))
        session.add_stage(MockStage(name="b"))
        session.add_stage(MockStage(name="c"))
        levels = session._topological_levels()
        assert len(levels) == 1
        assert set(levels[0]) == {"a", "b", "c"}

    def test_diamond_produces_three_levels(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="a"))
        session.add_stage(MockStage(name="b", depends_on=["a"]))
        session.add_stage(MockStage(name="c", depends_on=["a"]))
        session.add_stage(MockStage(name="d", depends_on=["b", "c"]))
        levels = session._topological_levels()
        assert len(levels) == 3
        assert levels[0] == ["a"]
        assert set(levels[1]) == {"b", "c"}
        assert levels[2] == ["d"]

    def test_linear_chain_all_single_levels(self):
        session = ResearchSession()
        session.add_stage(MockStage(name="a"))
        session.add_stage(MockStage(name="b", depends_on=["a"]))
        session.add_stage(MockStage(name="c", depends_on=["b"]))
        levels = session._topological_levels()
        assert levels == [["a"], ["b"], ["c"]]


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel_stages_all_execute(self):
        """Independent stages should all execute when run in parallel."""
        session = ResearchSession()
        stages = [MockStage(name="a"), MockStage(name="b"), MockStage(name="c")]
        for s in stages:
            session.add_stage(s)

        context = await session.run()
        assert all(s.executed for s in stages)
        assert set(context.results.keys()) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_diamond_parallel_execution(self):
        """Diamond DAG: a -> {b, c} -> d. b and c should run in parallel."""
        session = ResearchSession()
        a = MockStage(name="a")
        b = MockStage(name="b", depends_on=["a"])
        c = MockStage(name="c", depends_on=["a"])
        d = MockStage(name="d", depends_on=["b", "c"])
        for s in [a, b, c, d]:
            session.add_stage(s)

        context = await session.run()
        assert all(s.executed for s in [a, b, c, d])
        assert set(context.results.keys()) == {"a", "b", "c", "d"}


class TestSharedContext:
    def test_add_result_tracks_vulnerability(self):
        ctx = SharedContext()
        result = AnalysisResult(
            analysis_type="geometry",
            categories=["negation"],
            signals=[
                Signal(
                    type=SignalType.VULNERABILITY_DETECTED,
                    source="geometry",
                    category="negation",
                    data={"vulnerability_score": 0.9},
                )
            ],
        )
        ctx.add_result("geometry", result)
        assert "negation" in ctx.vulnerable_categories

    def test_add_result_tracks_stability(self):
        ctx = SharedContext()
        result = AnalysisResult(
            analysis_type="perturbation",
            categories=["polysemy"],
            signals=[
                Signal(
                    type=SignalType.STABILITY_MEASURED,
                    source="perturbation",
                    category="polysemy",
                    data={"mean_jaccard": 0.85},
                )
            ],
        )
        ctx.add_result("perturbation", result)
        assert "polysemy" in ctx.stable_categories
