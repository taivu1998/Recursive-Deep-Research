# Recursive Deep Research Agent - Source Package
# Lazy imports to avoid dependency issues during setup

__all__ = [
    "RecursiveAgent",
    "AgentState",
    "SubQuery",
    "ResearchPlan",
    "Assessment",
    "ConfigParser",
    "ResearchDataset",
    "Evaluator",
    "NaiveRAG",
]


def __getattr__(name):
    """Lazy import to avoid loading dependencies until needed."""
    if name == "RecursiveAgent":
        from src.model import RecursiveAgent
        return RecursiveAgent
    elif name in ("AgentState", "SubQuery", "ResearchPlan", "Assessment"):
        from src import schemas
        return getattr(schemas, name)
    elif name == "ConfigParser":
        from src.config_parser import ConfigParser
        return ConfigParser
    elif name == "ResearchDataset":
        from src.dataset import ResearchDataset
        return ResearchDataset
    elif name in ("Evaluator", "NaiveRAG"):
        from src import trainer
        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
