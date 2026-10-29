"""Train the upstream RGB safety-aware PFM model using explicit data/checkpoint paths."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pfp.training import main

if __name__ == '__main__':main()
