"""
Logic Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

Final Phase 2 Logic Agent that uses propositional logic and heuristic reasoning.
"""

import sys
import os
from typing import Tuple, List, Optional, Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment import GridWorld
from ai_core.knowledge_base import KnowledgeBase


class LogicAgent:
    """
    A rational agent that uses logical reasoning for navigation.
    
    This agent combines:
    1. Propositional logic knowledge base
    2. Forward chaining inference
    3. Goal-oriented heuristic reasoning
    4. Cycle avoidance with visited tracking
    """
    
    def __init__(self, environment: GridWorld):
        """
        Initialize the logic agent.
        
        Args:
            environment: The GridWorld environment
        """
        self.env = environment
        self.kb = KnowledgeBase()
        self.position = environment.start
        self.visited_positions: Set[Tuple[int, int]] = set([self.position])
        self.path_history: List[str] = []
        
        # Setup initial knowledge base rules
        self.setup_base_rules()
    
    def setup_base_rules(self):
        """Set up the fundamental inference rules for navigation."""
        # Clear any existing rules
        self.kb.rules.clear()
        
        # Basic safety and movement rules
        self.kb.add_rule(["Visited(X,Y)"], "Safe(X,Y)")
        self.kb.add_rule(["Free(X,Y)"], "Safe(X,Y)")
        
        # Movement capability inference
        self.kb.add_rule(["At(CX,CY)", "Safe(NX,NY)", "Adjacent(CX,CY,NX,NY)"], "CanMoveTo(NX,NY)")
        
        # Goal achievement
        self.kb.add_rule(["At(GX,GY)", "Goal(GX,GY)"], "AtGoal")
    
    def perceive_environment(self):
        """Perceive the current environment state and update knowledge base."""
        current_row, current_col = self.position
        goal_row, goal_col = self.env.goal
        
        # Clear facts but keep rules
        self.kb.clear_facts()
        
        # Add current state facts
        self.kb.tell(f"At({current_row},{current_col})")
        self.kb.tell(f"Visited({current_row},{current_col})")
        self.kb.tell(f"Goal({goal_row},{goal_col})")
        
        # Mark current position as visited
        self.visited_positions.add(self.position)
        
        # Check all adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dr, dc in directions:
            neighbor_row = current_row + dr
            neighbor_col = current_col + dc
            neighbor_pos = (neighbor_row, neighbor_col)
            
            # Add adjacency fact
            self.kb.tell(f"Adjacent({current_row},{current_col},{neighbor_row},{neighbor_col})")
            
            if self.env.is_valid(neighbor_pos):
                # Cell is traversable
                self.kb.tell(f"Free({neighbor_row},{neighbor_col})")
                self.kb.tell(f"Safe({neighbor_row},{neighbor_col})")
                
                if neighbor_pos not in self.visited_positions:
                    self.kb.tell(f"Unexplored({neighbor_row},{neighbor_col})")
            else:
                # Cell is blocked
                self.kb.tell(f"Obstacle({neighbor_row},{neighbor_col})")
        
        # Perform logical inference
        self.kb.infer()
    
    def reason_about_actions(self) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Use logical reasoning to determine possible actions.
        
        Returns:
            List of (direction, position) tuples for valid moves
        """
        current_row, current_col = self.position
        valid_moves = []
        
        # Check each possible direction
        directions = [
            ("up", (-1, 0)), ("down", (1, 0)),
            ("left", (0, -1)), ("right", (0, 1))
        ]
        
        for direction, (dr, dc) in directions:
            new_pos = (current_row + dr, current_col + dc)
            if self.env.is_valid(new_pos):
                valid_moves.append((direction, new_pos))
        
        return valid_moves
    
    def choose_best_action(self, valid_moves: List[Tuple[str, Tuple[int, int]]]) -> Optional[Tuple[str, Tuple[int, int]]]:
        """
        Choose the best action using heuristic reasoning.
        
        Args:
            valid_moves: List of possible (direction, position) moves
            
        Returns:
            The best (direction, position) to move to, or None if no moves
        """
        if not valid_moves:
            return None
        
        goal_row, goal_col = self.env.goal
        
        def manhattan_distance(pos):
            """Calculate Manhattan distance to goal."""
            return abs(pos[0] - goal_row) + abs(pos[1] - goal_col)
        
        # Separate unvisited and visited moves
        unvisited_moves = [move for move in valid_moves if move[1] not in self.visited_positions]
        visited_moves = [move for move in valid_moves if move[1] in self.visited_positions]
        
        # Prefer unvisited cells to encourage exploration
        if unvisited_moves:
            candidate_moves = unvisited_moves
        else:
            candidate_moves = visited_moves
        
        # Choose the move that minimizes distance to goal
        best_move = min(candidate_moves, key=lambda move: manhattan_distance(move[1]))
        
        return best_move
    
    def act(self) -> Tuple[bool, str]:
        """
        Execute one action cycle: perceive, reason, act.
        
        Returns:
            (success, description) of the action taken
        """
        # Perceive current environment
        self.perceive_environment()
        
        # Reason about possible actions
        valid_moves = self.reason_about_actions()
        
        if not valid_moves:
            return False, "No valid moves available"
        
        # Choose the best action
        best_action = self.choose_best_action(valid_moves)
        
        if best_action is None:
            return False, "Could not determine best action"
        
        direction, new_position = best_action
        
        # Execute the move
        old_position = self.position
        self.position = new_position
        self.visited_positions.add(new_position)
        
        action_description = f"Moved {direction} from {old_position} to {new_position}"
        self.path_history.append(action_description)
        
        return True, action_description
    
    def run_to_goal(self, max_steps: int = 200) -> Tuple[bool, int, List[str]]:
        """
        Run the agent until it reaches the goal or exceeds max steps.
        
        Args:
            max_steps: Maximum number of steps to attempt
            
        Returns:
            (success, steps_taken, action_history)
        """
        # Reset agent state
        self.position = self.env.start
        self.visited_positions = set([self.position])
        self.path_history = []
        
        steps = 0
        action_history = []
        
        print(f"🤖 Logic Agent starting at {self.position}")
        print(f"   Goal: {self.env.goal}, Max steps: {max_steps}")
        
        while steps < max_steps and self.position != self.env.goal:
            success, description = self.act()
            action_history.append(description)
            
            # Progress reporting
            if steps % 20 == 0:
                dist = abs(self.position[0] - self.env.goal[0]) + abs(self.position[1] - self.env.goal[1])
                print(f"   Step {steps}: {self.position} (distance: {dist})")
            
            if not success:
                print(f"   ⚠️  Stuck at step {steps}: {description}")
                break
            
            steps += 1
        
        success = self.position == self.env.goal
        
        if success:
            print(f"   ✅ Goal reached in {steps} steps!")
        else:
            print(f"   ❌ Failed to reach goal after {steps} steps")
            print(f"   Final position: {self.position}")
        
        return success, steps, action_history
    
    def get_knowledge_summary(self) -> dict:
        """Get a summary of the agent's current knowledge state."""
        return {
            "position": self.position,
            "goal": self.env.goal,
            "visited_positions": len(self.visited_positions),
            "knowledge_base_facts": len(self.kb.get_facts()),
            "knowledge_base_rules": len(self.kb.rules),
            "at_goal": self.position == self.env.goal
        }


# ============================================================================
# Testing and Demonstration
# ============================================================================

def demonstrate_logic_agent():
    """Demonstrate the logic agent on both test maps."""
    print("=" * 70)
    print("🤖 ROBOMIND - LOGIC AGENT DEMONSTRATION (PHASE 2)")
    print("=" * 70)
    
    test_maps = [
        ("Simple Map", "maps/simple.txt"),
        ("Maze Map", "maps/maze.txt")
    ]
    
    for map_name, map_file in test_maps:
        print(f"\n🗺️  {map_name.upper()}")
        print("-" * 50)
        
        # Load environment
        env = GridWorld()
        env.load_map(map_file)
        
        print(f"   Map: {env.width}x{env.height}")
        print(f"   Start: {env.start}, Goal: {env.goal}")
        print(f"   Obstacles: {(env.grid == 1).sum()}")
        
        # Create and run agent
        agent = LogicAgent(env)
        success, steps, history = agent.run_to_goal(300)
        
        # Display results
        print(f"\n   📊 RESULTS:")
        print(f"      Success: {'✅ YES' if success else '❌ NO'}")
        print(f"      Steps: {steps}")
        print(f"      Efficiency: {steps}/{(env.width + env.height) * 2} (actual/optimal ratio)")
        print(f"      Exploration: {len(agent.visited_positions)} cells visited")
        
        knowledge_summary = agent.get_knowledge_summary()
        print(f"      KB Facts: {knowledge_summary['knowledge_base_facts']}")
        print(f"      KB Rules: {knowledge_summary['knowledge_base_rules']}")
        
        if history and len(history) > 0:
            print(f"      First 5 moves: {[h.split(' ')[1] for h in history[:5]]}")
        
        print("-" * 50)


if __name__ == "__main__":
    demonstrate_logic_agent()
