"""
Evaluation Framework for Blackboard Agents

Provides tools for testing and scoring agent performance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .state import Blackboard, Status
from .core import Orchestrator

logger = logging.getLogger("blackboard.evals")


@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    goal: str
    expected_criteria: List[str] = field(default_factory=list)
    max_steps: int = 20
    timeout: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result from evaluating a single case."""
    case_id: str
    goal: str
    success: bool
    score: float = 0.0
    scores_by_criteria: Dict[str, float] = field(default_factory=dict)
    final_status: str = ""
    artifacts_count: int = 0
    steps_taken: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class EvalReport:
    """Report from running a full evaluation."""
    total_cases: int
    passed_cases: int
    failed_cases: int
    average_score: float
    average_steps: float
    average_time: float
    results: List[EvalResult] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": self.pass_rate,
            "average_score": self.average_score,
            "average_steps": self.average_steps,
            "average_time": self.average_time,
            "results": [
                {
                    "case_id": r.case_id,
                    "success": r.success,
                    "score": r.score,
                    "steps": r.steps_taken,
                    "time": r.execution_time
                }
                for r in self.results
            ]
        }


@runtime_checkable
class Judge(Protocol):
    """Protocol for evaluation judges."""
    
    async def score(
        self,
        result: Blackboard,
        criteria: List[str],
        goal: str
    ) -> Dict[str, Any]:
        """
        Score a blackboard result against criteria.
        
        Returns:
            Dict with 'score' (0-1), 'passed' (bool), 'reasoning' (str)
        """
        ...


class LLMJudge:
    """
    Uses an LLM to judge agent outputs.
    
    Compares the final blackboard state against specified criteria.
    
    Args:
        llm: LLM client for judging
        threshold: Score threshold for passing (0-1)
        
    Example:
        judge = LLMJudge(my_llm, threshold=0.7)
        score = await judge.score(
            result=blackboard,
            criteria=["Contains specific steps", "Includes time estimates"],
            goal="Create a project plan"
        )
    """
    
    JUDGE_PROMPT = '''You are evaluating an AI agent's output. Score how well it meets the criteria.

## Goal
{goal}

## Criteria to Evaluate
{criteria}

## Agent Output
{output}

## Instructions
1. Evaluate each criterion (1 = fully met, 0 = not met)
2. Provide overall score (0-1)
3. Explain your reasoning

Respond in JSON:
{{
    "criteria_scores": {{"criterion1": 0.8, "criterion2": 1.0}},
    "overall_score": 0.9,
    "passed": true,
    "reasoning": "Explanation..."
}}
'''
    
    def __init__(self, llm, threshold: float = 0.7):
        self.llm = llm
        self.threshold = threshold
    
    async def score(
        self,
        result: Blackboard,
        criteria: List[str],
        goal: str
    ) -> Dict[str, Any]:
        import json
        
        # Build output summary
        output_parts = [f"Status: {result.status.value}"]
        for artifact in result.artifacts:
            output_parts.append(f"\nArtifact ({artifact.type}):\n{artifact.content[:1000]}")
        
        output = "\n".join(output_parts)
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        
        prompt = self.JUDGE_PROMPT.format(
            goal=goal,
            criteria=criteria_text,
            output=output
        )
        
        # Get LLM judgment
        response = self.llm.generate(prompt)
        if asyncio.iscoroutine(response):
            response = await response
        
        # Parse response
        content = response.content if hasattr(response, 'content') else str(response)
        
        try:
            # Try to extract JSON
            json_match = content[content.find('{'):content.rfind('}')+1]
            data = json.loads(json_match)
            
            return {
                "score": data.get("overall_score", 0),
                "passed": data.get("passed", False),
                "scores_by_criteria": data.get("criteria_scores", {}),
                "reasoning": data.get("reasoning", "")
            }
        except:
            # Fallback
            logger.warning("Failed to parse judge response")
            return {
                "score": 0,
                "passed": False,
                "reasoning": f"Parse error: {content[:200]}"
            }


class RuleBasedJudge:
    """
    Rule-based judge using predefined checks.
    
    Faster than LLM judge, useful for automated testing.
    
    Args:
        rules: List of (name, check_fn) tuples
        threshold: Score threshold for passing
        
    Example:
        judge = RuleBasedJudge([
            ("has_artifact", lambda bb: len(bb.artifacts) > 0),
            ("completed", lambda bb: bb.status == Status.DONE),
        ])
    """
    
    def __init__(
        self,
        rules: List[tuple],
        threshold: float = 0.7
    ):
        self.rules = rules
        self.threshold = threshold
    
    async def score(
        self,
        result: Blackboard,
        criteria: List[str],
        goal: str
    ) -> Dict[str, Any]:
        scores = {}
        
        for name, check_fn in self.rules:
            try:
                scores[name] = 1.0 if check_fn(result) else 0.0
            except Exception as e:
                scores[name] = 0.0
                logger.warning(f"Rule '{name}' failed: {e}")
        
        overall = sum(scores.values()) / len(scores) if scores else 0
        
        return {
            "score": overall,
            "passed": overall >= self.threshold,
            "scores_by_criteria": scores,
            "reasoning": f"Passed {sum(1 for s in scores.values() if s > 0)}/{len(scores)} rules"
        }


class Evaluator:
    """
    Runs evaluations against a test dataset.
    
    Args:
        orchestrator: The orchestrator to evaluate
        judge: Judge for scoring results
        
    Example:
        evaluator = Evaluator(orchestrator, LLMJudge(my_llm))
        
        cases = [
            EvalCase(id="1", goal="Write a haiku", expected_criteria=["Has 3 lines"]),
            EvalCase(id="2", goal="Summarize an article", expected_criteria=["Under 100 words"]),
        ]
        
        report = await evaluator.run(cases)
        print(f"Pass rate: {report.pass_rate:.1%}")
    """
    
    def __init__(self, orchestrator: Orchestrator, judge: Judge):
        self.orchestrator = orchestrator
        self.judge = judge
    
    async def run(
        self,
        cases: List[EvalCase],
        parallel: bool = False
    ) -> EvalReport:
        """
        Run evaluation on all cases.
        
        Args:
            cases: List of test cases
            parallel: Whether to run cases in parallel
            
        Returns:
            EvalReport with results
        """
        if parallel:
            results = await asyncio.gather(
                *[self._run_case(case) for case in cases],
                return_exceptions=True
            )
            results = [
                r if not isinstance(r, Exception) else self._error_result(cases[i], r)
                for i, r in enumerate(results)
            ]
        else:
            results = []
            for case in cases:
                try:
                    result = await self._run_case(case)
                    results.append(result)
                except Exception as e:
                    results.append(self._error_result(case, e))
        
        # Build report
        passed = sum(1 for r in results if r.success)
        scores = [r.score for r in results]
        steps = [r.steps_taken for r in results]
        times = [r.execution_time for r in results]
        
        return EvalReport(
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=len(cases) - passed,
            average_score=sum(scores) / len(scores) if scores else 0,
            average_steps=sum(steps) / len(steps) if steps else 0,
            average_time=sum(times) / len(times) if times else 0,
            results=results
        )
    
    async def _run_case(self, case: EvalCase) -> EvalResult:
        """Run a single evaluation case."""
        start_time = time.time()
        
        try:
            # Run orchestrator
            result = await asyncio.wait_for(
                self.orchestrator.run(goal=case.goal, max_steps=case.max_steps),
                timeout=case.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Judge the result
            judgment = await self.judge.score(
                result=result,
                criteria=case.expected_criteria,
                goal=case.goal
            )
            
            return EvalResult(
                case_id=case.id,
                goal=case.goal,
                success=judgment["passed"],
                score=judgment["score"],
                scores_by_criteria=judgment.get("scores_by_criteria", {}),
                final_status=result.status.value,
                artifacts_count=len(result.artifacts),
                steps_taken=result.step_count,
                execution_time=execution_time,
                judge_reasoning=judgment.get("reasoning")
            )
            
        except asyncio.TimeoutError:
            return EvalResult(
                case_id=case.id,
                goal=case.goal,
                success=False,
                error="Timeout",
                execution_time=case.timeout
            )
    
    def _error_result(self, case: EvalCase, error: Exception) -> EvalResult:
        """Create error result for failed case."""
        return EvalResult(
            case_id=case.id,
            goal=case.goal,
            success=False,
            error=str(error)
        )
