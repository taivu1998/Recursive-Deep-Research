import logging
import json
import asyncio
import os
import pickle
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.model import RecursiveAgent
from src.dataset import ResearchDataset
from src.schemas import JudgeScore
from src.utils import setup_logger, save_artifact

logger = logging.getLogger("DeepResearch.Evaluator")


@dataclass
class RunMetrics:
    """Metrics collected from a single run."""
    question_id: int
    question: str
    answer: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0
    latency_seconds: float = 0.0
    num_iterations: int = 0
    trace: List[str] = field(default_factory=list)

    # Judge scores (filled after evaluation)
    completeness: int = 0
    factuality: int = 0
    coherence: int = 0
    judge_reasoning: str = ""


@dataclass
class EvaluationCheckpoint:
    """Checkpoint for resumable evaluation."""
    timestamp: str
    completed_question_ids: List[int]
    baseline_results: List[RunMetrics]
    agent_results: List[RunMetrics]
    current_phase: str  # 'baseline', 'agent', 'judging', 'complete'


class NaiveRAG:
    """
    Baseline: Simple Search + Answer (Zero-Shot RAG).
    No planning, no iteration, no critique.
    """

    def __init__(self, config: dict):
        self.config = config
        self.llm = ChatOpenAI(model=config['models']['worker'], temperature=0)
        self.search_tool = TavilySearchResults(max_results=config['agent']['search_max_results'])
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def run(self, query: str) -> Dict[str, Any]:
        """Single-shot search and answer."""
        # Search
        search_results = self.search_tool.invoke(query)

        # Generate answer
        prompt = f"""Answer the following question based on the search results.

Question: {query}

Search Results:
{json.dumps(search_results, indent=2)}

Provide a comprehensive answer:"""

        response = self.llm.invoke(prompt)

        # Track tokens
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self.prompt_tokens += response.usage_metadata.get('input_tokens', 0)
            self.completion_tokens += response.usage_metadata.get('output_tokens', 0)
            self.total_tokens = self.prompt_tokens + self.completion_tokens

        return {
            "answer": response.content,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class Evaluator:
    """
    The Evaluation Harness: Runs A/B tests between RecursiveAgent and NaiveRAG.
    Uses LLM-as-a-Judge for scoring. Supports checkpointing for resumable evaluation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("Evaluator", config['paths']['log_dir'])
        self.dataset = ResearchDataset(config['paths']['data_path'])

        # Checkpoint support
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "evaluation_checkpoint.pkl")

        # Initialize judge model
        self.judge = ChatOpenAI(model=config['models']['judge'], temperature=0)

        # Results storage
        self.agent_results: List[RunMetrics] = []
        self.baseline_results: List[RunMetrics] = []
        self.completed_ids: set = set()
        self.current_phase = 'baseline'

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint = EvaluationCheckpoint(
            timestamp=datetime.now().isoformat(),
            completed_question_ids=list(self.completed_ids),
            baseline_results=self.baseline_results,
            agent_results=self.agent_results,
            current_phase=self.current_phase
        )
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.logger.info(f"Checkpoint saved: {len(self.completed_ids)} questions completed")

    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists. Returns True if checkpoint was loaded."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.completed_ids = set(checkpoint.completed_question_ids)
            self.baseline_results = checkpoint.baseline_results
            self.agent_results = checkpoint.agent_results
            self.current_phase = checkpoint.current_phase
            self.logger.info(f"Checkpoint loaded from {checkpoint.timestamp}")
            self.logger.info(f"Phase: {self.current_phase}, Completed: {len(self.completed_ids)} questions")
            return True
        return False

    def train(self, resume: bool = True):
        """
        Main evaluation loop (named 'train' for ML convention compatibility).
        Runs both systems on the golden set and compares.
        Supports checkpointing for resumable evaluation.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Evaluation Harness")
        self.logger.info(f"Dataset size: {len(self.dataset)}")
        self.logger.info("=" * 60)

        # Try to resume from checkpoint
        if resume and self.load_checkpoint():
            self.logger.info("Resuming from checkpoint...")
        else:
            self.logger.info("Starting fresh evaluation...")
            self.completed_ids = set()
            self.baseline_results = []
            self.agent_results = []
            self.current_phase = 'baseline'

        # Run baseline
        if self.current_phase == 'baseline':
            self._run_baseline()
            self.current_phase = 'agent'
            self.save_checkpoint()

        # Run agent
        if self.current_phase == 'agent':
            self._run_agent()
            self.current_phase = 'judging'
            self.save_checkpoint()

        # Judge results
        if self.current_phase == 'judging':
            self._judge_results(self.baseline_results, "Baseline (NaiveRAG)")
            self._judge_results(self.agent_results, "RecursiveAgent")
            self.current_phase = 'complete'
            self.save_checkpoint()

        # Generate report
        report = self._generate_report()

        # Save artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_artifact(report, f"{self.config['paths']['checkpoint_dir']}/eval_report_{timestamp}.json")

        self._print_report(report)

        return report

    def _run_baseline(self):
        """Run NaiveRAG on all questions."""
        self.logger.info("\n--- Running Baseline (NaiveRAG) ---")

        for item in self.dataset:
            # Skip already completed questions
            if item['id'] in self.completed_ids:
                continue

            self.logger.info(f"Baseline Q{item['id']}: {item['question'][:50]}...")

            baseline = NaiveRAG(self.config)
            start_time = datetime.now()

            try:
                result = baseline.run(item['question'])
                latency = (datetime.now() - start_time).total_seconds()

                metrics = RunMetrics(
                    question_id=item['id'],
                    question=item['question'],
                    answer=result['answer'],
                    total_tokens=result['total_tokens'],
                    prompt_tokens=result['prompt_tokens'],
                    completion_tokens=result['completion_tokens'],
                    estimated_cost=self._estimate_cost(result, "baseline"),
                    latency_seconds=latency,
                    num_iterations=1,
                    trace=["search", "answer"]
                )
                self.baseline_results.append(metrics)
                self.completed_ids.add(item['id'])
                self.save_checkpoint()

            except Exception as e:
                self.logger.error(f"Baseline failed on Q{item['id']}: {e}")
                self.baseline_results.append(RunMetrics(
                    question_id=item['id'],
                    question=item['question'],
                    answer=f"ERROR: {str(e)}",
                ))

    def _run_agent(self):
        """Run RecursiveAgent on all questions."""
        self.logger.info("\n--- Running RecursiveAgent ---")

        agent = RecursiveAgent(self.config)
        agent_completed = {r.question_id for r in self.agent_results}

        for item in self.dataset:
            # Skip already completed questions
            if item['id'] in agent_completed:
                continue

            self.logger.info(f"Agent Q{item['id']}: {item['question'][:50]}...")

            start_time = datetime.now()
            inputs = {
                "original_query": item['question'],
                "plan": [],
                "results": {},
                "critique_count": 0,
                "messages": [],
                "draft_answer": ""
            }

            try:
                trace = []
                final_state = {}

                for event in agent.graph.stream(inputs):
                    for node, value in event.items():
                        trace.append(node)
                        final_state.update(value)

                latency = (datetime.now() - start_time).total_seconds()

                # Get token counts from agent (if tracked)
                total_tokens = getattr(agent, 'total_tokens', 0)

                metrics = RunMetrics(
                    question_id=item['id'],
                    question=item['question'],
                    answer=final_state.get('draft_answer', 'No answer generated'),
                    total_tokens=total_tokens,
                    estimated_cost=self._estimate_cost({'total_tokens': total_tokens}, "agent"),
                    latency_seconds=latency,
                    num_iterations=final_state.get('critique_count', 1),
                    trace=trace
                )
                self.agent_results.append(metrics)
                self.save_checkpoint()

            except Exception as e:
                self.logger.error(f"Agent failed on Q{item['id']}: {e}")
                self.agent_results.append(RunMetrics(
                    question_id=item['id'],
                    question=item['question'],
                    answer=f"ERROR: {str(e)}",
                ))

    def _judge_results(self, results: List[RunMetrics], system_name: str):
        """Use LLM-as-a-Judge to score answers."""
        self.logger.info(f"\n--- Judging {system_name} ---")

        judge_llm = self.judge.with_structured_output(JudgeScore)

        for metrics in results:
            if metrics.answer.startswith("ERROR"):
                metrics.completeness = 1
                metrics.factuality = 1
                metrics.coherence = 1
                metrics.judge_reasoning = "System error - no valid answer"
                continue

            prompt = f"""You are a Research Director evaluating an AI assistant's answer.

QUESTION: {metrics.question}

ANSWER TO EVALUATE:
{metrics.answer}

Score the answer on three dimensions (1-5 scale):
1. Completeness: Did it answer ALL parts of the question?
2. Factuality: Are the claims specific, verifiable, and grounded?
3. Coherence: Is the answer well-structured and easy to follow?

Be harsh but fair. A score of 3 is "acceptable", 4 is "good", 5 is "excellent"."""

            try:
                score = judge_llm.invoke(prompt)
                metrics.completeness = score.completeness
                metrics.factuality = score.factuality
                metrics.coherence = score.coherence
                metrics.judge_reasoning = score.reasoning
            except Exception as e:
                self.logger.error(f"Judge failed on Q{metrics.question_id}: {e}")
                metrics.judge_reasoning = f"Judge error: {str(e)}"

    def _estimate_cost(self, result: dict, system_type: str) -> float:
        """Estimate API cost based on token usage."""
        # Pricing (approximate, as of 2024)
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        # GPT-4o: $2.50/1M input, $10/1M output
        # Claude Sonnet: $3/1M input, $15/1M output

        total = result.get('total_tokens', 0)

        if system_type == "baseline":
            # Mostly GPT-4o-mini
            return (total / 1_000_000) * 0.375  # Avg of input/output
        else:
            # Mix of Sonnet (planner/critic) and GPT-4o-mini (worker)
            # Rough estimate: 60% Sonnet, 40% mini
            sonnet_cost = (total * 0.6 / 1_000_000) * 9  # Avg
            mini_cost = (total * 0.4 / 1_000_000) * 0.375
            return sonnet_cost + mini_cost

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        def avg_score(results: List[RunMetrics], attr: str) -> float:
            values = [getattr(r, attr) for r in results if getattr(r, attr) > 0]
            return sum(values) / len(values) if values else 0

        def total_metric(results: List[RunMetrics], attr: str) -> float:
            return sum(getattr(r, attr) for r in results)

        # Calculate win rates (agent vs baseline per question)
        wins = draws = losses = 0
        for agent_r, base_r in zip(self.agent_results, self.baseline_results):
            agent_total = agent_r.completeness + agent_r.factuality + agent_r.coherence
            base_total = base_r.completeness + base_r.factuality + base_r.coherence
            if agent_total > base_total:
                wins += 1
            elif agent_total == base_total:
                draws += 1
            else:
                losses += 1

        total_questions = len(self.agent_results)

        report = {
            "summary": {
                "total_questions": total_questions,
                "agent_win_rate": wins / total_questions if total_questions else 0,
                "agent_wins": wins,
                "draws": draws,
                "agent_losses": losses,
            },
            "baseline_metrics": {
                "avg_completeness": avg_score(self.baseline_results, "completeness"),
                "avg_factuality": avg_score(self.baseline_results, "factuality"),
                "avg_coherence": avg_score(self.baseline_results, "coherence"),
                "total_tokens": total_metric(self.baseline_results, "total_tokens"),
                "total_cost": total_metric(self.baseline_results, "estimated_cost"),
                "avg_latency": avg_score(self.baseline_results, "latency_seconds"),
            },
            "agent_metrics": {
                "avg_completeness": avg_score(self.agent_results, "completeness"),
                "avg_factuality": avg_score(self.agent_results, "factuality"),
                "avg_coherence": avg_score(self.agent_results, "coherence"),
                "total_tokens": total_metric(self.agent_results, "total_tokens"),
                "total_cost": total_metric(self.agent_results, "estimated_cost"),
                "avg_latency": avg_score(self.agent_results, "latency_seconds"),
                "avg_iterations": avg_score(self.agent_results, "num_iterations"),
            },
            "detailed_results": {
                "baseline": [asdict(r) for r in self.baseline_results],
                "agent": [asdict(r) for r in self.agent_results],
            }
        }

        # Calculate cost ratio
        base_cost = report["baseline_metrics"]["total_cost"]
        agent_cost = report["agent_metrics"]["total_cost"]
        report["summary"]["cost_ratio"] = agent_cost / base_cost if base_cost > 0 else 0

        return report

    def _print_report(self, report: Dict[str, Any]):
        """Print formatted evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT: RecursiveAgent vs NaiveRAG Baseline")
        print("=" * 70)

        s = report["summary"]
        print(f"\n{'SUMMARY':^70}")
        print("-" * 70)
        print(f"Total Questions: {s['total_questions']}")
        print(f"Agent Win Rate: {s['agent_win_rate']:.1%} ({s['agent_wins']}W / {s['draws']}D / {s['agent_losses']}L)")
        print(f"Cost Ratio (Agent/Baseline): {s['cost_ratio']:.2f}x")

        print(f"\n{'BASELINE (NaiveRAG)':^35} | {'RECURSIVE AGENT':^35}")
        print("-" * 70)

        b = report["baseline_metrics"]
        a = report["agent_metrics"]

        print(f"Completeness:  {b['avg_completeness']:.2f}/5              | Completeness:  {a['avg_completeness']:.2f}/5")
        print(f"Factuality:    {b['avg_factuality']:.2f}/5              | Factuality:    {a['avg_factuality']:.2f}/5")
        print(f"Coherence:     {b['avg_coherence']:.2f}/5              | Coherence:     {a['avg_coherence']:.2f}/5")
        print(f"Total Tokens:  {b['total_tokens']:,}               | Total Tokens:  {a['total_tokens']:,}")
        print(f"Est. Cost:     ${b['total_cost']:.4f}             | Est. Cost:     ${a['total_cost']:.4f}")
        print(f"Avg Latency:   {b['avg_latency']:.1f}s               | Avg Latency:   {a['avg_latency']:.1f}s")
        print(f"{'':35} | Avg Iterations: {a['avg_iterations']:.1f}")

        print("\n" + "=" * 70)
        print("Report saved to checkpoints/")
        print("=" * 70)
