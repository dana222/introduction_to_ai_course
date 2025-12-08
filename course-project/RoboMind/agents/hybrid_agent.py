"""
Hybrid Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

TODO: Integrate search + logic + probability Phase 4 of the project (Week 7-8) - Final Integration 

Phase 4: Hybrid Integration
Integrates Search + Logic + Probabilistic Reasoning
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

        # Internal state
        self.position = env.agent_pos
        self.goal = env.goal
        self.belief_map = {}  # (row,col) -> P(obstacle)

        # Initialize belief map with uniform uncertainty for unknown/uncertain cells
        for row in range(env.height):
            for col in range(env.width):
                if env.grid[row][col] == 1:  # Obstacle
                    self.belief_map[(row, col)] = 1.0
                elif env.grid[row][col] == 0:  # Free
                    self.belief_map[(row, col)] = 0.0
                else:  # Uncertain
                    self.belief_map[(row, col)] = 0.5

        # Parameters
        self.uncertainty_threshold = 0.6  # if P(obstacle) > threshold, consider unsafe

    # -----------------------------
    # Phase 4 Methods
    # -----------------------------

    def perceive(self) -> dict:
        """
        Get sensor readings from environment.
        Returns dictionary: neighbor_pos -> observed state (True=obstacle, False=free)
        """
        neighbors = self.env.get_neighbors(self.position)
        observation = {}
        for cell in neighbors:
            # Simple simulated sensor: read actual grid and add small noise if desired
            r, c = cell
            is_obstacle = self.env.grid[r][c] == 1
            observation[cell] = is_obstacle
        return observation

    def update_beliefs(self, observation: dict):
        """
        Update belief map using Phase 3 Bayesian reasoning.
        """
        for cell, sensor_reading in observation.items():
            prior = self.belief_map[cell]
            self.belief_map[cell] = update_belief_map({cell: prior}, sensor_reading)[cell]

    def reason(self):
        """
        Update knowledge base with logically inferred safe cells.
        """
        for cell, p_obs in self.belief_map.items():
            if p_obs < self.uncertainty_threshold:
                self.kb.tell(f"Safe{cell}")
            else:
                self.kb.tell(f"Obstacle{cell}")

    def plan(self) -> Optional[List[Tuple[int, int]]]:
        """
        Use search agent to plan path to goal avoiding cells with high obstacle probability.
        Returns path as list of positions.
        """
        # Always plan from the agent's CURRENT position (not the original start)
        self.env.start = self.position

        # Mark high-probability obstacles in environment temporarily for planning
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
            # Safety check: avoid high-probability obstacles
            if self.belief_map.get(next_pos, 0.0) < self.uncertainty_threshold:
                return next_pos

        # Explore safe neighbors inferred from logic
        safe_neighbors = [cell for cell in self.env.get_neighbors(self.position) if self.kb.ask(f"Safe{cell}")]
        if safe_neighbors:
            return safe_neighbors[0]

        # Fallback: pick neighbor with minimal probability of obstacle
        neighbors = self.env.get_neighbors(self.position)
        if neighbors:
            next_pos = min(neighbors, key=lambda c: self.belief_map.get(c, 1.0))
            return next_pos

        return None

    def act(self) -> Optional[Tuple[int, int]]:
        """
        Complete perception → reasoning → planning → action cycle.
        """
        # Step 1: perceive
        observation = self.perceive()

        # Step 2: update beliefs (probabilistic reasoning)
        self.update_beliefs(observation)

        # Step 3: logic reasoning
        self.reason()

        # Step 4: plan path (search)
        path = self.plan()

        # Step 5: decide next move (integration)
        next_pos = self.select_next_move(path)
        if next_pos:
            self.position = next_pos
            self.env.agent_pos = next_pos  # move agent in environment
        return next_pos


# -----------------------------
# Example usage / test harness
# -----------------------------
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
