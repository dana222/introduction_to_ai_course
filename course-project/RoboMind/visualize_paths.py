"""
Visualization helper for RoboMind paths.
"""

import pygame
import sys
from environment import GridWorld

def visualize_path(env, path, title="Path Visualization"):
    """
    Visualize a path in the environment.
    
    Args:
        env: GridWorld environment
        path: List of positions to highlight
        title: Window title
    """
    env.init_display()
    env.path = path
    env.visited = set(path)
    
    print(f"Visualizing path with {len(path)} steps")
    print(f"Press any key to step through, ESC to exit")
    
    step = 0
    running = True
    
    while running and step < len(path):
        env.agent_pos = path[step]
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    step = min(step + 1, len(path) - 1)
                elif event.key == pygame.K_LEFT:
                    step = max(step - 1, 0)
                elif event.key == pygame.K_SPACE:
                    step += 1
                    if step >= len(path):
                        running = False
        
        # Render
        env.screen.fill((255, 255, 255))
        
        # Draw grid
        for row in range(env.height):
            for col in range(env.width):
                pos = (row, col)
                if pos == env.start:
                    color = (76, 175, 80)  # Green
                elif pos == env.goal:
                    color = (244, 67, 54)  # Red
                elif pos in path[:step+1]:
                    color = (255, 235, 59)  # Yellow for visited path
                elif pos in path:
                    color = (255, 193, 7)   # Orange for future path
                elif env.grid[row][col] == 1:
                    color = (50, 50, 50)    # Black for obstacles
                else:
                    color = (255, 255, 255) # White for free space
                
                pygame.draw.rect(env.screen, color, 
                                (col * env.cell_size, 
                                 row * env.cell_size,
                                 env.cell_size, env.cell_size))
                pygame.draw.rect(env.screen, (200, 200, 200),
                                (col * env.cell_size,
                                 row * env.cell_size,
                                 env.cell_size, env.cell_size), 1)
        
        # Draw agent
        row, col = env.agent_pos
        center_x = col * env.cell_size + env.cell_size // 2
        center_y = row * env.cell_size + env.cell_size // 2
        pygame.draw.circle(env.screen, (33, 150, 243),
                          (center_x, center_y), env.cell_size // 3)
        
        # Draw info
        font = pygame.font.Font(None, 36)
        text = font.render(f"Step: {step}/{len(path)-1}", True, (0, 0, 0))
        env.screen.blit(text, (10, env.height * env.cell_size + 10))
        
        text = font.render(f"Position: {env.agent_pos}", True, (0, 0, 0))
        env.screen.blit(text, (10, env.height * env.cell_size + 50))
        
        pygame.display.flip()
        pygame.time.Clock().tick(10)  # Control speed
    
    env.close()

def compare_algorithms():
    """Compare different search algorithms visually."""
    from ai_core.search_algorithms import bfs, ucs, astar
    from agents.search_agent import SearchAgent
    
    # Create environment
    env = GridWorld(width=12, height=12, cell_size=40)
    
    # Add a challenging pattern
    for i in range(3, 9):
        env.add_obstacle(5, i)
        env.add_obstacle(i, 5)
    
    env.start = (1, 1)
    env.goal = (10, 10)
    
    # Test each algorithm
    algorithms = [
        ('BFS', lambda: bfs(env, env.start, env.goal)),
        ('UCS', lambda: ucs(env, env.start, env.goal)),
        ('A* Manhattan', lambda: astar(env, env.start, env.goal, 'manhattan')),
        ('A* Euclidean', lambda: astar(env, env.start, env.goal, 'euclidean'))
    ]
    
    print("Comparing search algorithms:")
    print("="*50)
    
    results = []
    for name, algo_func in algorithms:
        try:
            path, cost, expanded = algo_func()
            if path:
                results.append((name, path, cost, expanded))
                print(f"{name:<15} | Path: {len(path)-1:3d} | Cost: {cost:6.1f} | Expanded: {expanded:4d}")
            else:
                print(f"{name:<15} | No path found")
        except Exception as e:
            print(f"{name:<15} | Error: {e}")
    
    # Ask user which to visualize
    if results:
        print("\nWhich path would you like to visualize?")
        for i, (name, _, _, _) in enumerate(results):
            print(f"  {i+1}. {name}")
        
        choice = input("Enter number (or 'all'): ").strip()
        
        if choice.lower() == 'all':
            for name, path, _, _ in results:
                print(f"\nVisualizing {name}...")
                visualize_path(env, path, f"{name} Path")
        elif choice.isdigit() and 1 <= int(choice) <= len(results):
            name, path, _, _ = results[int(choice)-1]
            visualize_path(env, path, f"{name} Path")
        else:
            print("Invalid choice")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_algorithms()
    else:
        # Default: create a simple demo
        env = GridWorld(width=8, height=8, cell_size=60)
        env.add_obstacle(2, 2)
        env.add_obstacle(2, 3)
        env.add_obstacle(2, 4)
        env.add_obstacle(5, 5)
        env.start = (1, 1)
        env.goal = (6, 6)
        
        from ai_core.search_algorithms import astar
        path, cost, expanded = astar(env, env.start, env.goal)
        
        if path:
            visualize_path(env, path, "A* Search Path")
        else:
            print("No path found!")
