"""
Hybrid Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

TODO: Integrate search + logic + probability
Phase 4 of the project (Week 7-8) - Final Integration
"""

from typing import Tuple, List, Optional
from environment import GridWorld
from agents.search_agent import SearchAgent
from ai_core.knowledge_base import KnowledgeBase
from ai_core.bayes_reasoning import update_belief_map


class HybridAgent:
    """
    A rational agent that integrates search, logic, and probabilistic reasoning.
    """

    def __init__(self, env: GridWorld):
        self.env = env
        # Components
        self.search_agent = SearchAgent(env)
        self.kb = KnowledgeBase()
        self.position = env.agent_pos
        self.goal = env.goal
        self.belief_map = {}
        for row in range(env.height):
            for col in range(env.width):
                if env.grid[row][col] == 1:  
                    self.belief_map[(row, col)] = 1.0
                elif env.grid[row][col] == 0: 
                    self.belief_map[(row, col)] = 0.0
                else:  
                    self.belief_map[(row, col)] = 0.5
        self.uncertainty_threshold = 0.6 
    def perceive(self) -> dict:
        """
        Get sensor readings from environment.
        May be noisy - need probability!
        """
        neighbors = self.env.get_neighbors(self.position)
        observation = {}
        for cell in neighbors:
            r, c = cell
            is_obstacle = self.env.grid[r][c] == 1
            observation[cell] = is_obstacle
        return observation

    def update_beliefs(self, observation: dict):
        """
        Use Bayesian inference to handle uncertain sensor readings.
        """
        for cell, sensor_reading in observation.items():
            prior = self.belief_map[cell]
            self.belief_map[cell] = update_belief_map({cell: prior}, sensor_reading)[cell]

    def reason(self):
        """
        Use logic to infer safe moves and update knowledge base.
        """
        for cell, p_obs in self.belief_map.items():
            if p_obs < self.uncertainty_threshold:
                self.kb.tell(f"Safe{cell}")
            else:
                self.kb.tell(f"Obstacle{cell}")

    def plan(self) -> Optional[List[Tuple[int, int]]]:
        """
        Use search algorithms to plan path to goal.
        """
        self.env.start = self.position
        blocked_cells = [cell for cell, p in self.belief_map.items()
                         if p >= self.uncertainty_threshold]

        path, cost, expanded = self.search_agent.search('astar')
        return path

    def select_next_move(self, path: Optional[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """
        Decide next action based on integrated reasoning:
        1. Prefer path from search if exists
        2. Use logic to avoid unsafe cells
        3. If high uncertainty, pick neighbor with lowest P(obstacle)
        """
        if path and len(path) > 1:
            next_pos = path[1]
            if self.belief_map.get(next_pos, 0.0) < self.uncertainty_threshold:
                return next_pos
        safe_neighbors = [cell for cell in self.env.get_neighbors(self.position) if self.kb.ask(f"Safe{cell}")]
        if safe_neighbors:
            return safe_neighbors[0]
        neighbors = self.env.get_neighbors(self.position)
        if neighbors:
            next_pos = min(neighbors, key=lambda c: self.belief_map.get(c, 1.0))
            return next_pos

        return None

    def act(self) -> Optional[Tuple[int, int]]:
        """
        Integrate all reasoning techniques to decide next action.
        
        Strategy:
            1. If goal is visible and path is clear → use search
            2. If uncertain about obstacles → use probability
            3. If need to infer hidden info → use logic
        """
        observation = self.perceive()
        self.update_beliefs(observation)
        self.reason()
        path = self.plan()
        next_pos = self.select_next_move(path)
        if next_pos:
            self.position = next_pos
            self.env.agent_pos = next_pos
        return next_pos


# Example usage / test harness
if __name__ == "__main__":
    env = GridWorld(width=10, height=10)
    env.add_random_obstacles(15)
    env.start = (0, 0)
    env.goal = (9, 9)
    env.agent_pos = env.start
    agent = HybridAgent(env)
    step = 0
    max_steps = 50
    while agent.position != env.goal and step < max_steps:
        move = agent.act()
        print(f"Step {step}: Agent moves to {move}")
        step += 1
    if agent.position == env.goal:
        print("Agent reached the goal!")
    else:
        print("Agent did not reach the goal within step limit.")
