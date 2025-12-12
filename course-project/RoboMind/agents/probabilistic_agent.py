"""
Probabilistic Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

Implements a probabilistic reasoning agent using Bayes' rule.
"""

from environment import GridWorld
from typing import Tuple, List
from ai_core.bayes_reasoning import bayes_update, update_belief_map

class ProbabilisticAgent:
    """
    An agent that uses probabilistic reasoning to navigate the grid world.
    """

    def __init__(self, environment: GridWorld):
        """
        Initialize the probabilistic agent.

        Args:
            environment: The GridWorld environment
        """
        self.env = environment
        self.belief_map = [[1.0 / (environment.width * environment.height)
                            for _ in range(environment.width)]
                           for _ in range(environment.height)]
        self.path = []
        self.current_pos = environment.start

    def sense_and_update(self, observation):
        """
        Update belief map based on an observation using Bayes' rule.
        """
        self.belief_map = bayes_update(self.belief_map, observation, self.env)

    def plan_path(self, goal: Tuple[int, int]):
        """
        Plan a path to the goal using the current belief map.
        """
        self.path = update_belief_map(self.belief_map, self.current_pos, goal)

    def move_along_path(self):
        """
        Move the agent along the computed path (for visualization).
        """
        if not self.path:
            print("No path to follow!")
            return

        print(f"\n🤖 Moving along path ({len(self.path)} steps)...")
        for i, pos in enumerate(self.path):
            self.env.agent_pos = pos
            self.env.visited.add(pos)
            self.env.render()

            if self.env.is_goal(pos):
                print(f"✓ Goal reached at step {i+1}!")
                break
