"""
Probabilistic Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

TODO: Implement probabilistic reasoning with Bayes' rule
Phase 3 of the project (Week 5-6)
"""

from environment import GridWorld
from ai_core.bayes_reasoning import bayes_update, update_belief_map


class ProbabilisticAgent:
     """
    An agent that uses Bayesian reasoning to handle uncertainty.
    """
    def __init__(self, environment: GridWorld): 
        """Initialize the probabilistic agent."""
        self.env = environment 
        self.beliefs = {} 
        for row in range(self.env.height): 
            for col in range(self.env.width):
                self.beliefs[(row, col)] = 0.3 
        self.previousposition = None 
        
    def update_beliefs(self, sensor_reading, position):
        """Update beliefs using Bayes' rule."""
        neighborcells = self.env.get_neighbors(position) 
        updatingbeliefs = {}
        for cell in neighborcells:
            updatingbeliefs[cell] = self.beliefs[cell] 
        
        updatedbeliefs = update_belief_map(updatingbeliefs, sensor_reading, 0.9) 
        for cellposition, posterior in updatedbeliefs.items(): 
            self.beliefs[cellposition] = posterior


    def act(self):
        """Decide action based on probabilistic beliefs."""
        currentposition = self.env.agent_pos 
        nextmoves = self.env.get_neighbors(currentposition) 
        if not nextmoves:
            return currentposition
        moveoptions = [pos for pos in nextmoves if pos != self.previousposition]
        
        if len(moveoptions) == 0:
            moveoptions = nextmoves
        safest_move = moveoptions[0] 
        lowest_risk = self.beliefs.get(safest_move, 1.0) 
        for pos in moveoptions:
            currentrisk = self.beliefs.get(pos, 1.0)
            if currentrisk < lowest_risk: 
                lowest_risk = currentrisk
                safest_move = pos 
        
        self.previousposition = currentposition
        return safest_move
