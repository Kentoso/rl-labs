"""Render the best CarRacing-v3 checkpoint to MP4 videos.

Usage:
    python render_car_racing.py [checkpoint] [num_episodes]
"""

import sys

from dqn_car_racing import render


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_CarRacing-v3_best.pth"
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    render(checkpoint=checkpoint, num_episodes=num_episodes)
