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
    #setting up the agent and passing the enviroment
    def __init__(self, environment: GridWorld): 
        """Initialize the probabilistic agent."""
        self.env = environment # store the enviroment inside the agent
        self.beliefs = {}  # store the beliefs inside agent, the belief is a dictionary storing what the agent believes about each cell
        for row in range(self.env.height): # (self.env.height) is the no. of rows and we are looping around the grid 
            for col in range(self.env.width): # no. of columns
                self.beliefs[(row, col)] = 0.3 # assume that every cell has a 30% probability of obstacle
        self.previousposition = None # the agent did not move so has no prev position
        
    def update_beliefs(self, sensor_reading, position):
        """Update beliefs using Bayes' rule."""
        neighborcells = self.env.get_neighbors(position) # call func from enviroment.py
        updatingbeliefs = {}
        for cell in neighborcells:
            updatingbeliefs[cell] = self.beliefs[cell] # store the beliefs from self.beliefs into updatingbeliefs
        
        updatedbeliefs = update_belief_map(updatingbeliefs, sensor_reading, 0.9) # call func from bayes_reasoning where we are getting the posterior 
        # update the agents belief map with the new probab
        for cellposition, posterior in updatedbeliefs.items(): # store the posterior into the agent 
            self.beliefs[cellposition] = posterior

    # func where the agent moves
    def act(self):
        """Decide action based on probabilistic beliefs."""
        currentposition = self.env.agent_pos # get current position
        nextmoves = self.env.get_neighbors(currentposition) # call func from environment
        # if there are no places to move, stay in the current pos
        if not nextmoves:
            return currentposition
        # this removes the problem i encountered where the agent moves back and forth 
        moveoptions = [pos for pos in nextmoves if pos != self.previousposition]
        
        if len(moveoptions) == 0:
            moveoptions = nextmoves
        safest_move = moveoptions[0] # assuming first move is safe
        lowest_risk = self.beliefs.get(safest_move, 1.0) # the risk of the 1st move which is the least one to have an obstacle
        # go through each move and and get its probab of an obstacle
        for pos in moveoptions:
            currentrisk = self.beliefs.get(pos, 1.0) # the risk of this move
            if currentrisk < lowest_risk: # if the new risk is less than the 1st risk agent found before, move to it
                lowest_risk = currentrisk
                safest_move = pos # agent moves here
        
        self.previousposition = currentposition
        return safest_move
