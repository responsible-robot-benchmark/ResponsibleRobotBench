"""Train VLMbench policies with native/custom datasets or converted RGB data.

Example: python scripts/train_vlm.py +experiment=vlmbench_base data_root=datasets/native
Native data uses train/<task>/variation*/episodes and valid/<task>/variation*/episodes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfp.training import main

if __name__ == '__main__':
    main()
