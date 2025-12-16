import logging
import asyncio
from typing import Literal, List
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from src.schemas import AgentState, ResearchPlan, SubQuery, Assessment

logger = logging.getLogger("DeepResearch.Model")


class RecursiveAgent:
    """
    The 'Model' Architecture: A Recursive Plan-then-Search Agent.
    Implements inference-time compute with DAG-based planning and self-correction.
    """

    def __init__(self, config: dict):
        self.config = config
        self._init_models()
        self.graph = self._build_graph()
        self.search_tool = TavilySearchResults(max_results=config['agent']['search_max_results'])

        # Token tracking
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _init_models(self):
        # Planner & Critic: High reasoning (Sonnet)
        self.planner = ChatAnthropic(model=self.config['models']['planner'], temperature=0)
        # Worker: Cost efficient (GPT-4o-mini)
        self.worker = ChatOpenAI(model=self.config['models']['worker'], temperature=0)

    # --- Node Logic ---

    def _track_tokens(self, response):
        """Extract and accumulate token usage from LLM response."""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self.prompt_tokens += response.usage_metadata.get('input_tokens', 0)
            self.completion_tokens += response.usage_metadata.get('output_tokens', 0)
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def planner_node(self, state: AgentState):
        """Decomposes query into a DAG of sub-questions."""
        logger.info("--- PLANNER NODE ---")
        structured_llm = self.planner.with_structured_output(ResearchPlan)

        system_prompt = """You are a Research Architect. Your task is to decompose complex queries into a dependency graph (DAG) of sub-questions.

Rules:
1. Each sub-question should be atomic and searchable
2. Use 'dependencies' to specify which questions must be answered first
3. Questions with no dependencies can be executed in parallel
4. Provide clear reasoning for why each step is needed

Example for "Compare revenue of A vs B after event X":
- Step 1 (id=1): "When did event X occur?" (no deps)
- Step 2 (id=2): "What was A's revenue in the quarter after [date from 1]?" (deps=[1])
- Step 3 (id=3): "What was B's revenue in the quarter after [date from 1]?" (deps=[1])
- Step 4 (id=4): "How do these revenues compare?" (deps=[2,3])"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Decompose this query into a research plan:\n\n{state['original_query']}")
        ]

        plan = structured_llm.invoke(messages)

        logger.info(f"Generated {len(plan.steps)} sub-queries")
        for step in plan.steps:
            logger.info(f"  [{step.id}] {step.question} (deps: {step.dependencies})")

        return {
            "plan": plan.steps,
            "messages": [f"Plan generated with {len(plan.steps)} steps."]
        }

    def _execute_single_step(self, step: SubQuery, context: dict) -> tuple:
        """Execute a single search step and summarize results."""
        try:
            # Search
            search_results = self.search_tool.invoke(step.question)

            # Build context from dependencies
            dep_context = ""
            if step.dependencies and context:
                dep_context = "\n\nContext from previous steps:\n"
                for dep_id in step.dependencies:
                    if dep_id in context:
                        dep_context += f"- {context[dep_id]}\n"

            # Summarize with worker LLM
            prompt = f"""Synthesize the following search results to answer the question.
{dep_context}
Question: {step.question}

Search Results:
{search_results}

Provide a concise, factual summary that directly answers the question:"""

            response = self.worker.invoke(prompt)
            self._track_tokens(response)

            return (step.id, response.content)
        except Exception as e:
            logger.error(f"Error executing step {step.id}: {e}")
            return (step.id, f"Error: {str(e)}")

    def executor_node(self, state: AgentState):
        """Executes steps where dependencies are met, using parallel execution."""
        logger.info("--- EXECUTOR NODE ---")
        plan = state['plan']
        results = state.get('results', {})
        completed_ids = set(results.keys())

        # 1. Identify runnable steps (dependencies satisfied, not yet completed)
        runnable = []
        for step in plan:
            if step.id not in completed_ids:
                if all(dep in completed_ids for dep in step.dependencies):
                    runnable.append(step)

        if not runnable:
            logger.info("No runnable steps found (waiting on dependencies)")
            return {"messages": ["Waiting on dependent steps to complete"]}

        logger.info(f"Found {len(runnable)} runnable steps: {[s.id for s in runnable]}")

        # 2. Execute in parallel using ThreadPoolExecutor
        new_results = {}
        if len(runnable) > 1:
            logger.info("Executing steps in parallel...")
            with ThreadPoolExecutor(max_workers=min(len(runnable), 5)) as executor:
                futures = [
                    executor.submit(self._execute_single_step, step, results)
                    for step in runnable
                ]
                for future in futures:
                    step_id, summary = future.result()
                    new_results[step_id] = summary
                    logger.info(f"  Completed step {step_id}")
        else:
            # Single step, no need for threading overhead
            step = runnable[0]
            logger.info(f"Executing step {step.id}: {step.question[:50]}...")
            step_id, summary = self._execute_single_step(step, results)
            new_results[step_id] = summary

        return {"results": new_results, "messages": [f"Executed {len(new_results)} steps in parallel"]}

    def aggregator_node(self, state: AgentState):
        """Synthesizes all results into a comprehensive answer."""
        logger.info("--- AGGREGATOR NODE ---")

        # Build structured context from all completed steps
        context_parts = []
        for step in state['plan']:
            result = state['results'].get(step.id, "Not yet completed")
            context_parts.append(f"Sub-question {step.id}: {step.question}\nFindings: {result}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a Research Analyst synthesizing findings into a comprehensive answer.

Original Query: {state['original_query']}

Research Findings:
{context}

Instructions:
1. Synthesize all findings into a coherent, well-structured answer
2. Cite specific facts from the research where relevant
3. Acknowledge any gaps or uncertainties
4. Be comprehensive but concise

Provide your answer:"""

        response = self.planner.invoke(prompt)
        self._track_tokens(response)

        return {"draft_answer": response.content}

    def critic_node(self, state: AgentState):
        """The 'System 2' critic - evaluates answer sufficiency and identifies gaps."""
        logger.info("--- CRITIC NODE ---")
        structured_critic = self.planner.with_structured_output(Assessment)

        prompt = f"""You are a harsh Research Director reviewing a draft answer.

ORIGINAL QUERY: {state['original_query']}

DRAFT ANSWER:
{state['draft_answer']}

RESEARCH STEPS COMPLETED:
{[f"{s.id}: {s.question}" for s in state['plan']]}

Evaluate critically:
1. Does this answer FULLY address the original query?
2. Are there any factual gaps or unverified claims?
3. Is the answer specific enough with concrete data/dates/numbers?

If the answer is insufficient, provide specific new sub-questions that would fill the gaps.
Be harsh but constructive."""

        assessment = structured_critic.invoke(prompt)

        logger.info(f"Critic assessment: sufficient={assessment.is_sufficient}")
        logger.info(f"Feedback: {assessment.feedback[:100]}...")

        updates = {
            "critique_count": state.get('critique_count', 0) + 1,
            "messages": [f"Critic [{state.get('critique_count', 0) + 1}]: {assessment.feedback[:100]}..."]
        }

        if not assessment.is_sufficient and assessment.new_sub_questions:
            # Dynamic Plan Extension
            current_max = max((s.id for s in state['plan']), default=0)
            new_steps = []
            for i, q in enumerate(assessment.new_sub_questions):
                new_steps.append(SubQuery(
                    id=current_max + 1 + i,
                    question=q,
                    dependencies=[],  # Independent for immediate execution
                    reasoning=f"Critic feedback: {assessment.feedback}"
                ))
            updates["plan"] = new_steps  # Will be appended due to operator.add
            logger.info(f"Added {len(new_steps)} new steps to plan")

        return updates

    def executor_routing(self, state: AgentState) -> Literal["executor", "aggregator"]:
        """Route after executor: continue executing or aggregate results."""
        completed = set(state['results'].keys())
        all_steps = set(s.id for s in state['plan'])

        # Check if there are more runnable steps (dependencies might now be satisfied)
        runnable = []
        for step in state['plan']:
            if step.id not in completed:
                if all(dep in completed for dep in step.dependencies):
                    runnable.append(step)

        if runnable:
            logger.info(f"More steps can now run: {[s.id for s in runnable]}")
            return "executor"

        logger.info("All runnable steps completed. Moving to aggregation.")
        return "aggregator"

    def critic_routing(self, state: AgentState) -> Literal["end", "executor"]:
        """Route after critic: end or continue with new steps."""
        # Circuit breaker: prevent infinite loops
        if state['critique_count'] >= self.config['agent']['max_loops']:
            logger.info(f"Max iterations ({self.config['agent']['max_loops']}) reached. Ending.")
            return "end"

        # Check if there are uncompleted steps (critic may have added new ones)
        completed = set(state['results'].keys())
        all_steps = set(s.id for s in state['plan'])
        pending = all_steps - completed

        if not pending:
            logger.info("All steps completed and critic satisfied. Ending.")
            return "end"

        logger.info(f"Pending steps: {pending}. Continuing to executor.")
        return "executor"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("aggregator", self.aggregator_node)
        workflow.add_node("critic", self.critic_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")

        # After executor: either continue executing or aggregate
        workflow.add_conditional_edges(
            "executor",
            self.executor_routing,
            {"executor": "executor", "aggregator": "aggregator"}
        )

        workflow.add_edge("aggregator", "critic")

        # After critic: either end or loop back for more research
        workflow.add_conditional_edges(
            "critic",
            self.critic_routing,
            {"end": END, "executor": "executor"}
        )

        return workflow.compile()