"""
Search Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

phase 1 
"""

from typing import Tuple, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ai_core.search_algorithms import bfs, ucs, astar

class SearchAgent:
    def __init__(self, env):
        self.env = env
        self.path = None
        self.current_step = 0
    
    def search(self, algorithm: str, heuristic: str = 'manhattan') -> Tuple[Optional[List], float, int]:
        start = self.env.start
        goal = self.env.goal
        
        print(f"Searching with {algorithm.upper()} from {start} to {goal}...")
        
        if algorithm == 'bfs':
            path, cost, expanded = bfs(self.env, start, goal)
        elif algorithm == 'ucs':
            path, cost, expanded = ucs(self.env, start, goal)
        elif algorithm == 'astar':
            path, cost, expanded = astar(self.env, start, goal, heuristic)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.path = path
        self.current_step = 0
        
        if path:
            self.env.path = path
            self.env.expanded = expanded
        else:
            self.env.path = []
            self.env.expanded = expanded
            
        return path, cost, expanded
    
    def get_next_move(self) -> Optional[Tuple[int, int]]:
        if self.path is None or self.current_step >= len(self.path):
            return None
        
        next_pos = self.path[self.current_step]
        self.current_step += 1
        return next_pos
    
    def has_path(self) -> bool:
        return self.path is not None and len(self.path) > 0
    
    def is_at_goal(self) -> bool:
        if not self.has_path():
            return False
        return self.current_step >= len(self.path)
    
    def reset(self):
        self.path = None
        self.current_step = 0

from collections import deque

def bidirectional_search(env, start, goal):
    if start == goal:
        return [start], 0, 0
    front_start = {start}
    front_goal = {goal}
    parent_start = {start: None}
    parent_goal = {goal: None}

    expanded = 0

    while front_start and front_goal:
        if len(front_start) <= len(front_goal):
            frontier = front_start
            parents = parent_start
            other_parents = parent_goal
            direction = "forward"
        else:
            frontier = front_goal
            parents = parent_goal
            other_parents = parent_start
            direction = "backward"

        next_frontier = set()

        for node in frontier:
            expanded += 1

            for neighbor in env.get_neighbors(node):
                if neighbor not in parents:
                    parents[neighbor] = node
                    next_frontier.add(neighbor)
                if neighbor in other_parents:
                    return _reconstruct_bidirectional_path(
                        parent_start, parent_goal, neighbor
                    ), len(_reconstruct_bidirectional_path(parent_start, parent_goal, neighbor)) - 1, expanded

        if direction == "forward":
            front_start = next_frontier
        else:
            front_goal = next_frontier
    return None, float('inf'), expanded


def _reconstruct_bidirectional_path(parent_start, parent_goal, meet):
    path_start = []
    node = meet
    while node is not None:
        path_start.append(node)
        node = parent_start[node]
    path_start.reverse()
    path_goal = []
    node = parent_goal[meet]
    while node is not None:
        path_goal.append(node)
        node = parent_goal[node]
    return path_start + path_goal
