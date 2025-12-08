"""
Hybrid Agent - RoboMind Project
SE444 - AI Course Project

Phase 4: Hybrid Integration
Search + Logic + Probabilistic Reasoning combined
"""

from typing import Tuple,List,Optional
from environment import GridWorld
from agents.search_agent import SearchAgent
from ai_core.knowledge_base import KnowledgeBase
from ai_core.bayes_reasoning import update_belief_map


class HybridAgent:
    def __init__(self,env:GridWorld):
        self.env=env
        self.searcher=SearchAgent(env)
        self.kb=KnowledgeBase()
        self.pos=env.agent_pos
        self.target=env.goal
        self.beliefs={}
        for y in range(env.height):
            for x in range(env.width):
                if env.grid[y][x]==1:
                    self.beliefs[(y,x)]=1.0
                elif env.grid[y][x]==0:
                    self.beliefs[(y,x)]=0.0
                else:
                    self.beliefs[(y,x)]=0.5
        self.uncert_thresh=0.6
        
    def sense(self)->dict:
        nearby=self.env.get_neighbors(self.pos)
        obs={}
        for spot in nearby:
            r,c=spot
            is_blocked=self.env.grid[r][c]==1
            obs[spot]=is_blocked
        return obs
        
    def update_belief(self,observation:dict):
        for loc,sensor_val in observation.items():
            prior=self.beliefs[loc]
            self.beliefs[loc]=update_belief_map({loc:prior},sensor_val)[loc]
            
    def reason_logic(self):
        for cell,prob in self.beliefs.items():
            if prob<self.uncert_thresh:
                self.kb.tell(f"Safe{cell}")
            else:
                self.kb.tell(f"Obstacle{cell}")

    def make_plan(self)->Optional[List[Tuple[int,int]]]:
        self.env.start=self.pos
        blocked=[loc for loc,p in self.beliefs.items() if p>=self.uncert_thresh]
        route,cost,exp=self.searcher.search('astar')
        return route

    def pick_move(self,path:Optional[List[Tuple[int,int]]])->Optional[Tuple[int,int]]:
        if path and len(path)>1:
            next_cell=path[1]
            if self.beliefs.get(next_cell,0.0)<self.uncert_thresh:
                return next_cell
        safe_neighbors=[cell for cell in self.env.get_neighbors(self.pos) if self.kb.ask(f"Safe{cell}")]
        if safe_neighbors:
            return safe_neighbors[0]
        neighbors=self.env.get_neighbors(self.pos)
        if neighbors:
            next_pos=min(neighbors,key=lambda c:self.beliefs.get(c,1.0))
            return next_pos
        return None

    def act(self)->Optional[Tuple[int,int]]:
        obs=self.sense()
        self.update_belief(obs)
        self.reason_logic()
        route=self.make_plan()
        nxt=self.pick_move(route)
        if nxt:
            self.pos=nxt
            self.env.agent_pos=nxt
        return nxt


# Example run
if __name__=="__main__":
    env=GridWorld(width=10,height=10)
    env.add_random_obstacles(15)
    env.start=(0,0)
    env.goal=(9,9)
    env.agent_pos=env.start

    robot=HybridAgent(env)

    steps=0
    max_steps=50
    while robot.pos!=env.goal and steps<max_steps:
        mov=robot.act()
        print(f"Step {steps}: moved to {mov}")
        steps+=1

    if robot.pos==env.goal:
        print("Reached target!")
    else:
        print("Didnt reach goal in time.")
