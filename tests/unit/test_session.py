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
