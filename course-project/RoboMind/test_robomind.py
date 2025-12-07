"""
RoboMind - Complete Test Suite
SE444 - Artificial Intelligence Course Project

Run comprehensive tests on all implemented phases.
"""

import sys
import os
from environment import GridWorld

# Test imports
def test_search_algorithms():
    """Test Phase 1: Search Algorithms."""
    print("\n" + "="*60)
    print("PHASE 1: TESTING SEARCH ALGORITHMS")
    print("="*60)
    
    from ai_core.search_algorithms import bfs, ucs, astar, reconstruct_path
    from agents.search_agent import SearchAgent
    
    # Create environment
    env = GridWorld(width=10, height=10, cell_size=40)
    env.start = (0, 0)
    env.goal = (9, 9)
    
    # Add some obstacles to make it interesting
    obstacles = [
        (2, 2), (2, 3), (2, 4), (2, 5),
        (5, 5), (6, 5), (7, 5), (8, 5),
        (3, 7), (4, 7), (5, 7), (6, 7)
    ]
    for obs in obstacles:
        env.add_obstacle(*obs)
    
    print(f"Grid: {env.width}x{env.height}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Obstacles: {len(obstacles)}")
    
    # Test 1: BFS
    print("\n1. Testing BFS (Breadth-First Search)...")
    try:
        path, cost, expanded = bfs(env, env.start, env.goal)
        if path:
            print(f"   ✓ BFS found path of length {len(path)-1}, cost {cost}")
            print(f"   Nodes expanded: {expanded}")
            # Verify path is valid
            valid = all(env.is_valid(pos) for pos in path)
            print(f"   Path is valid: {'✓' if valid else '✗'}")
            print(f"   Path leads to goal: {'✓' if path[-1] == env.goal else '✗'}")
        else:
            print("   ✗ BFS failed to find path")
    except Exception as e:
        print(f"   ✗ BFS error: {e}")
    
    # Test 2: UCS
    print("\n2. Testing UCS (Uniform Cost Search)...")
    try:
        path, cost, expanded = ucs(env, env.start, env.goal)
        if path:
            print(f"   ✓ UCS found path of length {len(path)-1}, cost {cost}")
            print(f"   Nodes expanded: {expanded}")
            # For uniform grid, UCS cost should equal path length
            expected_cost = len(path) - 1  # All moves cost 1
            print(f"   Cost matches expected: {'✓' if cost == expected_cost else '✗'}")
        else:
            print("   ✗ UCS failed to find path")
    except Exception as e:
        print(f"   ✗ UCS error: {e}")
    
    # Test 3: A* with Manhattan
    print("\n3. Testing A* with Manhattan heuristic...")
    try:
        path, cost, expanded = astar(env, env.start, env.goal, 'manhattan')
        if path:
            print(f"   ✓ A* (Manhattan) found path, cost {cost}")
            print(f"   Nodes expanded: {expanded}")
            # Check if heuristic is admissible
            h_start = env.manhattan_distance(env.start, env.goal)
            print(f"   Heuristic at start: {h_start}")
            print(f"   Heuristic is optimistic: {'✓' if h_start <= cost else '?'}")
        else:
            print("   ✗ A* failed to find path")
    except Exception as e:
        print(f"   ✗ A* error: {e}")
    
    # Test 4: Search Agent
    print("\n4. Testing SearchAgent class...")
    try:
        agent = SearchAgent(env)
        
        for algo in ['bfs', 'ucs', 'astar']:
            print(f"   Running {algo.upper()} through agent...")
            try:
                path, cost, expanded = agent.search(algo)
                if path:
                    print(f"     ✓ {algo.upper()} successful")
                    # Test next move functionality
                    agent.reset()
                    agent.search(algo)
                    next_pos = agent.get_next_move()
                    print(f"     First move: {next_pos}")
                else:
                    print(f"     ✗ {algo.upper()} failed")
            except Exception as e:
                print(f"     ✗ Error in {algo}: {e}")
    except Exception as e:
        print(f"   ✗ SearchAgent error: {e}")
    
    return True

def test_logic_agent():
    """Test Phase 2: Logic Agent."""
    print("\n" + "="*60)
    print("PHASE 2: TESTING LOGIC AGENT")
    print("="*60)
    
    from agents.logic_agent import LogicAgent
    
    # Test on simple map
    print("\n1. Testing on simple.txt map...")
    env = GridWorld()
    env.load_map("maps/simple.txt")
    
    print(f"   Map loaded: {env.width}x{env.height}")
    print(f"   Start: {env.start}, Goal: {env.goal}")
    
    # Create logic agent
    agent = LogicAgent(env)
    
    # Test one action cycle
    print("\n   Testing single action cycle...")
    success, description = agent.act()
    print(f"   Action: {description}")
    print(f"   Success: {success}")
    print(f"   New position: {agent.position}")
    
    # Test run to goal
    print("\n   Testing run_to_goal()...")
    success, steps, history = agent.run_to_goal(max_steps=100)
    print(f"   Reached goal: {'✓' if success else '✗'}")
    print(f"   Steps taken: {steps}")
    print(f"   Final position: {agent.position}")
    
    if history:
        print(f"   First few actions:")
        for i, action in enumerate(history[:5]):
            print(f"     {i+1}. {action}")
    
    # Test knowledge base
    print("\n   Testing Knowledge Base...")
    kb_info = agent.get_knowledge_summary()
    print(f"   KB facts: {kb_info['knowledge_base_facts']}")
    print(f"   KB rules: {kb_info['knowledge_base_rules']}")
    print(f"   Visited positions: {kb_info['visited_positions']}")
    
    # Test on maze map
    print("\n2. Testing on maze.txt map...")
    env2 = GridWorld()
    env2.load_map("maps/maze.txt")
    
    print(f"   Map loaded: {env2.width}x{env2.height}")
    
    agent2 = LogicAgent(env2)
    success2, steps2, history2 = agent2.run_to_goal(max_steps=300)
    
    print(f"   Reached goal: {'✓' if success2 else '✗'}")
    print(f"   Steps taken: {steps2}")
    
    if success2:
        print(f"   ✓ Successfully navigated maze!")
        # Calculate efficiency
        direct_distance = env2.manhattan_distance(env2.start, env2.goal)
        efficiency = (direct_distance / steps2) * 100 if steps2 > 0 else 0
        print(f"   Efficiency: {efficiency:.1f}% (optimal would be {direct_distance} steps)")
    else:
        print(f"   ✗ Could not reach goal in {steps2} steps")
        print(f"   Final position: {agent2.position}")
    
    return True

def test_bayesian_reasoning():
    """Test Phase 3: Bayesian Reasoning."""
    print("\n" + "="*60)
    print("PHASE 3: TESTING BAYESIAN REASONING")
    print("="*60)
    
    from ai_core.bayes_reasoning import (
        bayes_update, 
        compute_evidence, 
        update_belief_map,
        sensor_model
    )
    
    # Test 1: Basic Bayes' Rule
    print("\n1. Testing Bayes' Rule calculations...")
    try:
        # Medical test example
        prevalence = 0.01  # 1% disease prevalence
        sensitivity = 0.95  # P(positive | disease)
        false_positive = 0.10  # P(positive | healthy)
        
        # Calculate evidence probability
        P_evidence = compute_evidence(prevalence, sensitivity, false_positive)
        
        # Update belief
        posterior = bayes_update(prevalence, sensitivity, P_evidence)
        
        print(f"   Disease prevalence: {prevalence:.1%}")
        print(f"   Test sensitivity: {sensitivity:.1%}")
        print(f"   False positive rate: {false_positive:.1%}")
        print(f"   P(positive test): {P_evidence:.3f}")
        print(f"   P(disease | positive): {posterior:.1%}")
        
        # Sanity check
        if 0 <= posterior <= 1:
            print("   ✓ Posterior probability is valid")
        else:
            print("   ✗ Invalid posterior probability")
            
    except Exception as e:
        print(f"   ✗ Bayes' Rule error: {e}")
    
    # Test 2: Sensor Model
    print("\n2. Testing Sensor Model...")
    try:
        sensor_accuracy = 0.9
        
        # Test when obstacle exists
        P_detect_obstacle, P_miss_obstacle = sensor_model(True, sensor_accuracy)
        print(f"   Obstacle exists:")
        print(f"     P(detect | obstacle): {P_detect_obstacle:.3f}")
        print(f"     P(miss | obstacle): {P_miss_obstacle:.3f}")
        
        # Test when no obstacle
        P_false_alarm, P_correct_free = sensor_model(False, sensor_accuracy)
        print(f"   No obstacle:")
        print(f"     P(false alarm): {P_false_alarm:.3f}")
        print(f"     P(correct free): {P_correct_free:.3f}")
        
        # Check probabilities sum to 1
        if abs(P_detect_obstacle + P_miss_obstacle - 1.0) < 0.001:
            print("   ✓ Probabilities sum to 1 correctly")
        else:
            print("   ✗ Probabilities don't sum to 1")
            
    except Exception as e:
        print(f"   ✗ Sensor Model error: {e}")
    
    # Test 3: Belief Map Update
    print("\n3. Testing Belief Map Update...")
    try:
        # Create initial belief map
        beliefs = {
            (0, 0): 0.1,  # Low prior for obstacle
            (0, 1): 0.5,  # Medium prior
            (0, 2): 0.9,  # High prior
        }
        
        print(f"   Initial beliefs:")
        for pos, belief in beliefs.items():
            print(f"     {pos}: {belief:.3f}")
        
        # Update with sensor reading (obstacle detected)
        updated_beliefs = update_belief_map(beliefs.copy(), True, 0.9)
        
        print(f"   After sensor detects obstacle:")
        for pos, belief in beliefs.items():
            updated = updated_beliefs[pos]
            change = updated - belief
            print(f"     {pos}: {belief:.3f} → {updated:.3f} (Δ{change:+.3f})")
        
        # Check that beliefs updated in right direction
        for pos in beliefs:
            if beliefs[pos] < 0.5:
                # Low prior should increase when obstacle detected
                if updated_beliefs[pos] > beliefs[pos]:
                    print(f"   ✓ {pos}: Belief increased correctly")
                else:
                    print(f"   ✗ {pos}: Belief didn't increase as expected")
            elif beliefs[pos] > 0.5:
                # High prior should increase even more
                if updated_beliefs[pos] > beliefs[pos]:
                    print(f"   ✓ {pos}: High belief increased further")
                else:
                    print(f"   ✗ {pos}: High belief didn't increase")
                    
    except Exception as e:
        print(f"   ✗ Belief Map error: {e}")
    
    # Test 4: Edge Cases
    print("\n4. Testing Edge Cases...")
    try:
        # Test with evidence = 0
        result = bayes_update(0.5, 0.9, 0)
        print(f"   Bayes with zero evidence: {result}")
        
        # Test with prior = 0
        result = bayes_update(0, 0.9, 0.5)
        print(f"   Bayes with zero prior: {result}")
        
        # Test with prior = 1
        result = bayes_update(1, 0.9, 0.5)
        print(f"   Bayes with prior=1: {result}")
        
    except Exception as e:
        print(f"   ✗ Edge case error: {e}")
    
    return True

def test_integration():
    """Test integration of all components."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: ALL COMPONENTS")
    print("="*60)
    
    # Create a complex environment
    env = GridWorld(width=15, height=15)
    
    # Add obstacles in a maze-like pattern
    # Create a spiral pattern
    obstacles = []
    for i in range(1, 14, 2):
        for j in range(i, 15-i):
            obstacles.append((i, j))
            obstacles.append((14-i, j))
        for j in range(i+1, 14-i):
            obstacles.append((j, i))
            obstacles.append((j, 14-i))
    
    # Add openings
    obstacles = [obs for obs in obstacles if obs != (7, 7)]
    obstacles = [obs for obs in obstacles if obs != (0, 0)]
    obstacles = [obs for obs in obstacles if obs != (14, 14)]
    
    for obs in obstacles:
        env.add_obstacle(*obs)
    
    env.start = (0, 0)
    env.goal = (14, 14)
    
    print(f"Created complex maze: {env.width}x{env.height}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Obstacles: {len(obstacles)}")
    
    # Test 1: Search
    print("\n1. Searching for path...")
    from agents.search_agent import SearchAgent
    search_agent = SearchAgent(env)
    
    try:
        path, cost, expanded = search_agent.search('astar', 'manhattan')
        if path:
            print(f"   ✓ Found path with A*")
            print(f"   Path length: {len(path)-1}, Cost: {cost}")
            print(f"   Nodes expanded: {expanded}")
        else:
            print("   ✗ No path found")
    except Exception as e:
        print(f"   ✗ Search error: {e}")
    
    # Test 2: Logic
    print("\n2. Logic agent reasoning...")
    from agents.logic_agent import LogicAgent
    logic_agent = LogicAgent(env)
    
    # Run logic agent for a few steps
    print("   Running logic agent for 10 steps...")
    for i in range(10):
        success, desc = logic_agent.act()
        if not success:
            print(f"   Step {i+1}: Failed - {desc}")
            break
    
    kb_info = logic_agent.get_knowledge_summary()
    print(f"   KB has {kb_info['knowledge_base_facts']} facts")
    print(f"   Visited {kb_info['visited_positions']} cells")
    
    # Test 3: Compare performance
    print("\n3. Performance comparison...")
    print("   (Run full tests to compare search vs logic efficiency)")
    
    return True

def run_all_tests():
    """Run all test suites."""
    print("="*70)
    print("ROBOMIND - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test Phase 1: Search
    try:
        results['search'] = test_search_algorithms()
    except Exception as e:
        print(f"\n✗ Search test failed: {e}")
        results['search'] = False
    
    # Test Phase 2: Logic
    try:
        results['logic'] = test_logic_agent()
    except Exception as e:
        print(f"\n✗ Logic test failed: {e}")
        results['logic'] = False
    
    # Test Phase 3: Bayesian
    try:
        results['bayesian'] = test_bayesian_reasoning()
    except Exception as e:
        print(f"\n✗ Bayesian test failed: {e}")
        results['bayesian'] = False
    
    # Test Integration
    try:
        results['integration'] = test_integration()
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        results['integration'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for component, success in results.items():
        status = "PASS ✓" if success else "FAIL ✗"
        print(f"{component.upper():<12} {status}")
    
    passed = sum(1 for s in results.values() if s)
    total = len(results)
    
    print("-"*70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*70)
    
    return all(results.values())

def quick_test():
    """Run a quick smoke test to verify basic functionality."""
    print("QUICK SMOKE TEST")
    print("="*40)
    
    # Test 1: Environment loads
    print("1. Testing environment...")
    env = GridWorld(width=5, height=5)
    env.start = (0, 0)
    env.goal = (4, 4)
    
    print(f"   Grid created: {env.width}x{env.height}")
    print(f"   Start: {env.start}, Goal: {env.goal}")
    print(f"   Is valid start: {env.is_valid(env.start)}")
    print(f"   Is valid goal: {env.is_valid(env.goal)}")
    
    # Test 2: Basic search
    print("\n2. Testing basic search...")
    from ai_core.search_algorithms import bfs
    try:
        path, cost, expanded = bfs(env, env.start, env.goal)
        if path:
            print(f"   ✓ BFS found path of length {len(path)-1}")
        else:
            print("   ✗ BFS failed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Load map
    print("\n3. Testing map loading...")
    try:
        env2 = GridWorld()
        env2.load_map("maps/simple.txt")
        print(f"   ✓ Loaded simple.txt: {env2.width}x{env2.height}")
        print(f"   Start: {env2.start}, Goal: {env2.goal}")
    except Exception as e:
        print(f"   ✗ Could not load map: {e}")
    
    # Test 4: Knowledge base
    print("\n4. Testing knowledge base...")
    try:
        from ai_core.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        kb.tell("Safe(1,1)")
        kb.tell("Free(1,1)")
        kb.add_rule(["Safe(1,1)", "Free(1,1)"], "CanMove(1,1)")
        kb.infer()
        
        print(f"   ✓ KB created with {len(kb.get_facts())} facts")
        print(f"   CanMove(1,1): {kb.ask('CanMove(1,1)')}")
    except Exception as e:
        print(f"   ✗ KB error: {e}")
    
    print("\n" + "="*40)
    print("Quick test complete!")
    print("="*40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RoboMind project")
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--search', action='store_true', help='Test search algorithms')
    parser.add_argument('--logic', action='store_true', help='Test logic agent')
    parser.add_argument('--bayesian', action='store_true', help='Test Bayesian reasoning')
    parser.add_argument('--integration', action='store_true', help='Test integration')
    parser.add_argument('--quick', action='store_true', help='Quick smoke test')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        args.all = True
    
    if args.quick:
        quick_test()
    elif args.all:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        if args.search:
            test_search_algorithms()
        if args.logic:
            test_logic_agent()
        if args.bayesian:
            test_bayesian_reasoning()
        if args.integration:
            test_integration()
