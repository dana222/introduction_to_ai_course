"""
Logic Agent - RoboMind Project
SE444 - AI Course Project

Phase 2 :
"""

import sys
import os
from typing import Tuple,List,Optional,Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment import GridWorld
from ai_core.knowledge_base import KnowledgeBase


class LogicAgent:
    def __init__(self,env:GridWorld):
        self.env=env
        self.kb=KnowledgeBase()
        self.pos=env.start
        self.visited:Set[Tuple[int,int]]=set([self.pos])
        self.path_history:List[str]=[]
        self.setup_rules()

    def setup_rules(self):
        self.kb.rules.clear()
        self.kb.add_rule(["Visited(X,Y)"],"Safe(X,Y)")
        self.kb.add_rule(["Free(X,Y)"],"Safe(X,Y)")
        self.kb.add_rule(
            ["At(CX,CY)","Safe(NX,NY)","Adjacent(CX,CY,NX,NY)"],
            "CanMoveTo(NX,NY)"
        )
        self.kb.add_rule(
            ["At(GX,GY)","Goal(GX,GY)"],
            "AtGoal"
        )

    def act(self)->Tuple[bool,str]:
        self.perceive()

        valid_moves=self.get_valid_moves()
        if not valid_moves:
            return False,"No moves."

        choice=self.choose_action(valid_moves)
        if not choice:
            return False,"Cant choose."

        direction,newpos=choice
        oldpos=self.pos
        self.pos=newpos
        self.visited.add(newpos)

        desc=f"Moved {direction} from {oldpos} to {newpos}"
        self.path_history.append(desc)

        return True,desc

     def choose_action(self,valid_moves:List[Tuple[str,Tuple[int,int]]])->Optional[Tuple[str,Tuple[int,int]]]:
        if not valid_moves:
            return None
        goal_r,goal_c=self.env.goal
         def manhattan(p):
            return abs(p[0]-goal_r)+abs(p[1]-goal_c)
        unvistd=[m for m in valid_moves if m[1] not in self.visited]
        candidates=unvistd if unvistd else valid_moves
        return min(candidates,key=lambda mv:manhattan(mv[1]))

    def get_valid_moves(self)->List[Tuple[str,Tuple[int,int]]]:
        r,c=self.pos
        moves=[]
        directions=[
            ("up",(-1,0)),
            ("down",(1,0)),
            ("left",(0,-1)),
            ("right",(0,1))
        ]

        for name,(dr,dc) in directions:
            newpos=(r+dr,c+dc)
            if self.env.is_valid(newpos):
                moves.append((name,newpos))

        return moves

    def run_to_goal(self,max_steps:int=200)->Tuple[bool,int,List[str]]:
        self.pos=self.env.start
        self.visited=set([self.pos])
        self.path_history=[]
        steps=0
        hist=[]

        while steps<max_steps and self.pos!=self.env.goal:
            success,desc=self.act()
            hist.append(desc)

            if not success:
                break

            steps+=1

        return (self.pos==self.env.goal),steps,hist

     def perceive(self):
        """Sense grid and load facts."""
        r,c=self.pos
        goal_r,goal_c=self.env.goal
        self.kb.clear_facts()
        self.kb.tell(f"At({r},{c})")
        self.kb.tell(f"Visited({r},{c})")
        self.kb.tell(f"Goal({goal_r},{goal_c})")
        self.visited.add(self.pos)
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]
        for dr,dc in dirs:
            nr,nc=r+dr,c+dc
            pos=(nr,nc)
            self.kb.tell(f"Adjacent({r},{c},{nr},{nc})")
            if self.env.is_valid(pos):
                self.kb.tell(f"Free({nr},{nc})")
                self.kb.tell(f"Safe({nr},{nc})")

                if pos not in self.visited:
                    self.kb.tell(f"Unexplored({nr},{nc})")
            else:
                self.kb.tell(f"Obstacle({nr},{nc})")

        self.kb.infer()


    def get_summary(self)->dict:
        return {
            "pos":self.pos,
            "goal":self.env.goal,
            "visited_count":len(self.visited),
            "facts_count":len(self.kb.get_facts()),
            "rules_count":len(self.kb.rules),
            "at_goal":self.pos==self.env.goal
        }
