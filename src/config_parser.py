import argparse
import yaml
import os
from typing import Dict, Any

class ConfigParser:
    """
    Parses YAML configuration files and allows command-line overrides.
    """
    def __init__(self, config_path: str = None):
        self.args = self._parse_args()
        # Use CLI config path if provided, else default
        self.config_path = self.args.config or config_path
        self.config = self._load_config()
        self._override_config()

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Recursive Deep Research Agent")
        parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to YAML config')
        parser.add_argument('--query', type=str, help='Override query for inference')
        parser.add_argument('--device', type=str, help='Override device (cpu/cuda)')
        parser.add_argument('--max_loops', type=int, help='Override max recursion loops')
        return parser.parse_args()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _override_config(self):
        """Overrides YAML values with CLI arguments."""
        if self.args.query:
            self.config['query_override'] = self.args.query
        if self.args.device:
            self.config['device'] = self.args.device
        if self.args.max_loops:
            self.config['agent']['max_loops'] = self.args.max_loops
            
    def get_config(self) -> Dict[str, Any]:
        return self.config