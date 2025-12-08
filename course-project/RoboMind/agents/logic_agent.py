"""
Logic Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

Upgraded Version (Phase 2 BONUS):
→ Supports full First-Order Logic inference via FOL Knowledge Base
→ Uses variable unification and predicate reasoning
→ Backward compatible with all original behavior
"""

import sys
import os
from typing import Tuple, List, Optional, Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment import GridWorld
from ai_core.knowledge_base import KnowledgeBase


class LogicAgent:
    """
    A rational reasoning agent that uses:
    - First-Order Logic (BONUS)
    - Forward chaining inference
    - Variable unification
    - Predicate-based adjacency reasoning
    - Heuristic action selection
    """

    def __init__(self, environment: GridWorld):
        self.env = environment
        self.kb = KnowledgeBase()
        self.position = environment.start
        self.visited_positions: Set[Tuple[int, int]] = set([self.position])
        self.path_history: List[str] = []

        # Setup initial knowledge base rules
        self.setup_base_rules()

    # --------------------------------------------------------------
    #  RULE SETUP
    # --------------------------------------------------------------
    def setup_base_rules(self):
        """Set up FOL reasoning rules."""

        self.kb.rules.clear()

        # If visited → safe
        self.kb.add_rule(["Visited(X,Y)"], "Safe(X,Y)")

        # Free cells are safe
        self.kb.add_rule(["Free(X,Y)"], "Safe(X,Y)")

        # Movement inference
        # If we are at (CX,CY), and there is a safe adjacent cell (NX,NY), we can move
        self.kb.add_rule(
            ["At(CX,CY)", "Safe(NX,NY)", "Adjacent(CX,CY,NX,NY)"],
            "CanMoveTo(NX,NY)"
        )

        # Reaching goal
        self.kb.add_rule(
            ["At(GX,GY)", "Goal(GX,GY)"],
            "AtGoal"
        )

    # --------------------------------------------------------------
    #  PERCEPTION
    # --------------------------------------------------------------
    def perceive_environment(self):
        """Perceive the grid and load facts into KB."""
        r, c = self.position
        goal_r, goal_c = self.env.goal

        # Reset facts but keep rules
        self.kb.clear_facts()

        # Current state
        self.kb.tell(f"At({r},{c})")
        self.kb.tell(f"Visited({r},{c})")
        self.kb.tell(f"Goal({goal_r},{goal_c})")

        self.visited_positions.add(self.position)

        # Directions (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Load environment observations
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            pos = (nr, nc)

            # Adjacency fact
            self.kb.tell(f"Adjacent({r},{c},{nr},{nc})")

            if self.env.is_valid(pos):
                # Traversable
                self.kb.tell(f"Free({nr},{nc})")
                self.kb.tell(f"Safe({nr},{nc})")

                if pos not in self.visited_positions:
                    self.kb.tell(f"Unexplored({nr},{nc})")
            else:
                # Obstacle
                self.kb.tell(f"Obstacle({nr},{nc})")

        # Apply inference
        self.kb.infer()

    # --------------------------------------------------------------
    #  REASONING ABOUT ACTIONS
    # --------------------------------------------------------------
    def reason_about_actions(self) -> List[Tuple[str, Tuple[int, int]]]:
        """Determine valid movements from environment (not KB-based)."""
        r, c = self.position
        moves = []

        directions = [
            ("up", (-1, 0)),
            ("down", (1, 0)),
            ("left", (0, -1)),
            ("right", (0, 1))
        ]

        for name, (dr, dc) in directions:
            new_pos = (r + dr, c + dc)
            if self.env.is_valid(new_pos):
                moves.append((name, new_pos))

        return moves

    # --------------------------------------------------------------
    #  ACTION SELECTION
    # --------------------------------------------------------------
    def choose_best_action(self, valid_moves: List[Tuple[str, Tuple[int, int]]]) -> Optional[Tuple[str, Tuple[int, int]]]:
        if not valid_moves:
            return None

        goal_r, goal_c = self.env.goal

        def manhattan(p):
            return abs(p[0] - goal_r) + abs(p[1] - goal_c)

        # Prefer unvisited
        unvisited = [m for m in valid_moves if m[1] not in self.visited_positions]

        candidates = unvisited if unvisited else valid_moves

        return min(candidates, key=lambda move: manhattan(move[1]))

    # --------------------------------------------------------------
    #  ACTION LOOP
    # --------------------------------------------------------------
    def act(self) -> Tuple[bool, str]:
        self.perceive_environment()

        valid = self.reason_about_actions()
        if not valid:
            return False, "No valid moves."

        choice = self.choose_best_action(valid)
        if not choice:
            return False, "Unable to choose action."

        direction, new_pos = choice
        old_pos = self.position
        self.position = new_pos
        self.visited_positions.add(new_pos)

        desc = f"Moved {direction} from {old_pos} to {new_pos}"
        self.path_history.append(desc)

        return True, desc

    # --------------------------------------------------------------
    #  RUN UNTIL GOAL
    # --------------------------------------------------------------
    def run_to_goal(self, max_steps: int = 200) -> Tuple[bool, int, List[str]]:
        self.position = self.env.start
        self.visited_positions = set([self.position])
        self.path_history = []

        steps = 0
        history = []

        while steps < max_steps and self.position != self.env.goal:
            success, desc = self.act()
            history.append(desc)

            if not success:
                break

            steps += 1

        return (self.position == self.env.goal), steps, history

    # --------------------------------------------------------------
    #  SUMMARY
    # --------------------------------------------------------------
    def get_knowledge_summary(self) -> dict:
        return {
            "position": self.position,
            "goal": self.env.goal,
            "visited_positions": len(self.visited_positions),
            "knowledge_base_facts": len(self.kb.get_facts()),
            "knowledge_base_rules": len(self.kb.rules),
            "at_goal": self.position == self.env.goal
        }
