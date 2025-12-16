import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_parser import ConfigParser
from src.utils import seed_everything
from src.trainer import Evaluator

def main():
    load_dotenv()
    parser = ConfigParser()
    config = parser.get_config()
    
    seed_everything(config['seed'])
    
    evaluator = Evaluator(config)
    evaluator.train()

if __name__ == "__main__":
    main()