"""
Probabilistic Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

TODO: Implement probabilistic reasoning with Bayes' rule
Phase 3 of the project (Week 5-6)
"""

from environment import GridWorld
from ai_core.bayes_reasoning import update_belief_map

class ProbabilisticAgent:
    def __init__(self, environment: GridWorld):
        self.env = environment
        self.beliefs = {}
        
        for row in range(self.env.height):
            for col in range(self.env.width):
                self.beliefs[(row, col)] = 0.2
        
        self.previous_position = None
        
    def update_beliefs(self, sensor_reading: bool, current_position):
        neighbor_cells = self.env.get_neighbors(current_position)
        
        beliefs_to_update = {}
        for cell in neighbor_cells:
            beliefs_to_update[cell] = self.beliefs[cell]
        
        updated_beliefs = update_belief_map(beliefs_to_update, sensor_reading, 0.9)
        
        for cell_pos, new_prob in updated_beliefs.items():
            self.beliefs[cell_pos] = new_prob
    
    def act(self):
        current_pos = self.env.agent_pos
        possible_moves = self.env.get_neighbors(current_pos)
        
        if not possible_moves:
            return current_pos
        
        move_options = [pos for pos in possible_moves if pos != self.previous_position]
        
        if len(move_options) == 0:
            move_options = possible_moves
        
        safest_move = move_options[0]
        lowest_risk = self.beliefs.get(safest_move, 1.0)
        for pos in move_options:
            risk = self.beliefs.get(pos, 1.0)
            if risk < lowest_risk:
                lowest_risk = risk
                safest_move = pos
        
        self.previous_position = current_pos
        
        return safest_move
