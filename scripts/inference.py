import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_parser import ConfigParser
from src.utils import setup_logger, seed_everything
from src.model import RecursiveAgent


def main():
    load_dotenv()
    parser = ConfigParser()
    config = parser.get_config()

    seed_everything(config['seed'])
    logger = setup_logger("Inference")

    query = config.get('query_override') or "Explain the architecture of Mamba."

    logger.info(f"Initializing Agent... (Planner: {config['models']['planner']})")
    agent = RecursiveAgent(config)

    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}\n")

    inputs = {
        "original_query": query,
        "plan": [],
        "results": {},
        "critique_count": 0,
        "messages": [],
        "draft_answer": ""
    }

    # Stream events to visualize the "Thinking" (Trace)
    # Accumulate state properly across all nodes
    accumulated_state = inputs.copy()

    print("TRACE (Thinking Process):")
    print("-" * 60)

    for event in agent.graph.stream(inputs):
        for node, value in event.items():
            print(f"\n[NODE: {node}]")

            # Accumulate state updates
            for key, val in value.items():
                if key == "plan" and isinstance(val, list):
                    # Plans are appended (operator.add)
                    accumulated_state["plan"] = accumulated_state.get("plan", []) + val
                elif key == "results" and isinstance(val, dict):
                    # Results are merged (operator.ior)
                    accumulated_state["results"] = {**accumulated_state.get("results", {}), **val}
                elif key == "messages" and isinstance(val, list):
                    # Messages are appended
                    accumulated_state["messages"] = accumulated_state.get("messages", []) + val
                else:
                    accumulated_state[key] = val

            # Show relevant logs
            if "messages" in value and value["messages"]:
                print(f"  Log: {value['messages'][-1]}")
            if "plan" in value:
                print(f"  Steps added: {len(value['plan'])}")
            if "results" in value:
                print(f"  Results obtained: {list(value['results'].keys())}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(accumulated_state.get('draft_answer', 'No answer generated'))

    print("\n" + "-" * 60)
    print("STATISTICS:")
    print(f"  Total iterations: {accumulated_state.get('critique_count', 0)}")
    print(f"  Total sub-questions: {len(accumulated_state.get('plan', []))}")
    print(f"  Total tokens used: {agent.total_tokens:,}")
    print("-" * 60)


if __name__ == "__main__":
    main()