"""
Search Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project

Agent that uses search algorithms to navigate the grid world.
"""

from typing import Tuple, List, Optional
import sys
import os

# Add the parent directory to path to import ai_core
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_core.search_algorithms import bfs, ucs, astar


class SearchAgent:
    """
    An AI agent that uses search algorithms to find paths in the grid world.
    """
    
   def __init__(self, grid_env):
    """
    Setup the pathfinding agent with its environment.
    
    Parameters:
        grid_env: The GridWorld instance to navigate
    """
    self.world = grid_env
    self.planned_route = None
    self.route_position = 0

def is_active(self) -> bool:
    """
    Check if the agent has a valid route to follow.
    
    Returns:
        True if route exists and has steps remaining
    """
    return self.planned_route is not None and len(self.planned_route) > 0

def reached_destination(self) -> bool:
    """
    Determine if agent has completed its route.
    
    Returns:
        True if at final position in route
    """
    if not self.is_active():
        return False
    return self.route_position >= len(self.planned_route)

def compute_route(self, method: str, distance_heuristic: str = 'manhattan') -> Tuple[Optional[List], float, int]:
    """
    Calculate optimal path using selected search algorithm.
    
    Parameters:
        method: Algorithm choice ('bfs', 'ucs', 'astar')
        distance_heuristic: 'manhattan' or 'euclidean' (A* only)
    
    Returns:
        route: Sequence of positions from origin to target
        total_cost: Sum of movement costs
        explored_count: Number of nodes processed during search
    """
    origin = self.world.start
    target = self.world.goal
    
    print(f"Computing route via {method.upper()} from {origin} to {target}...")
    
    # Execute appropriate search strategy
    if method == 'bfs':
        route, total_cost, explored_count = bfs(self.world, origin, target)
    elif method == 'ucs':
        route, total_cost, explored_count = ucs(self.world, origin, target)
    elif method == 'astar':
        route, total_cost, explored_count = astar(self.world, origin, target, distance_heuristic)
    else:
        raise ValueError(f"Unsupported search method: {method}")
    
    # Store computed route for execution
    self.planned_route = route
    self.route_position = 0
    
    # Update environment state for display
    if route:
        self.world.path = route
        self.world.expanded = explored_count
    else:
        self.world.path = []
        self.world.expanded = explored_count
        
    return route, total_cost, explored_count

def advance(self) -> Optional[Tuple[int, int]]:
    """
    Move agent to next position in planned route.
    
    Returns:
        Next coordinate (row, column) or None if route completed
    """
    if self.planned_route is None or self.route_position >= len(self.planned_route):
        return None
    
    next_coordinate = self.planned_route[self.route_position]
    self.route_position += 1
    return next_coordinate

def restart(self):
    """
    Clear current route and reset agent to initial state.
    """
    self.planned_route = None
    self.route_position = 0
