import sys
import os

# Alias for evaluate.py to satisfy ML tooling conventions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.evaluate import main

if __name__ == "__main__":
    main()