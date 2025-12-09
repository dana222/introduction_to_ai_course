"""
Bayesian Reasoning Module
SE444 - Artificial Intelligence Course Project

TODO: Implement Bayesian belief updates for handling uncertainty
Phase 3 (Week 5-6)
"""

from typing import Dict, Tuple


def bayes_update(prior: float, likelihood: float, evidence: float) -> float:
    if evidence == 0:
        return prior #bc prior is the belief before seeing the evidence, we can't say return 1/0 here bc it means that the hypothesis is T/F and we don't know that without evidence 
    posterior = (likelihood * prior) / evidence
    return posterior

def compute_evidence(prior: float, likelihood_h: float, likelihood_not_h: float) -> float:
    p_notH = 1 - prior
    Probability_Evidence = (likelihood_h * prior) + (likelihood_not_h * p_notH)
    return Probability_Evidence
                          

def update_belief_map(belief_map: Dict[Tuple[int, int], float],
                      sensor_reading: bool,
                      sensor_accuracy: float = 0.9) -> Dict[Tuple[int, int], float]:
    
    for key, value in belief_map.items(): #to loop over the cells (keys) and prior (values) in the belief map
        prior = value
        if sensor_reading == True:
            likelihood_h = sensor_accuracy
            likelihood_not_h = 1 - sensor_accuracy
        else:
            likelihood_h = 1 - sensor_accuracy
            likelihood_not_h = sensor_accuracy

        p_notH = 1 - prior
        Probability_Evidence = (likelihood_h * prior) + (likelihood_not_h * p_notH)
        posterior = (likelihood_h * prior) / Probability_Evidence 
        belief_map[key] = posterior
    return belief_map


def sensor_model(actual_state: bool, sensor_accuracy: float = 0.9) -> Tuple[float, float]:
    #the probability the sensor will say there is an obstacle or not
    if actual_state == True: # obstacle exists
        return sensor_accuracy, 1 - sensor_accuracy #the probability the sensor says T, the probability the sensor says F
    else:  # no obstacle
        return 1 - sensor_accuracy, sensor_accuracy #the probability the sensor says T, the probability the sensor says F


# ============================================================================
# Testing Code
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Testing Bayesian Reasoning")
    print("=" * 60 + "\n")
    
    print("Example: Medical diagnosis")
    print("-" * 40)
    print("Disease prevalence: 1% (P(Disease) = 0.01)")
    print("Test accuracy: 95% (P(+|Disease) = 0.95)")
    print("False positive: 10% (P(+|Healthy) = 0.10)")
    print("\nPatient tests positive. What's the probability they have the disease?")
    
    try:
        # Prior
        P_disease = 0.01
        P_healthy = 1 - P_disease
        
        # Likelihood
        P_pos_given_disease = 0.95
        P_pos_given_healthy = 0.10
        
        # Evidence
        P_pos = compute_evidence(P_disease, P_pos_given_disease, P_pos_given_healthy)
        
        # Posterior
        P_disease_given_pos = bayes_update(P_disease, P_pos_given_disease, P_pos)
        
        print(f"\nResult: P(Disease|+) = {P_disease_given_pos:.1%}")
        print("(Surprisingly low despite positive test!)")
        
    except NotImplementedError:
        print("\n⚠️  Bayes' rule not implemented yet!")
    
    print("\n" + "=" * 60)
    print("  Example: Robot Sensor")
    print("=" * 60)
    print("\nRobot sensor is 90% accurate")
    print("Prior belief cell has obstacle: 30%")
    print("Sensor detects obstacle")
    print("\nWhat's updated belief?")
    
    try:
        P_obstacle = 0.30
        P_detect_if_obstacle = 0.90
        P_detect_if_free = 0.10
        
        P_detect = compute_evidence(P_obstacle, P_detect_if_obstacle, P_detect_if_free)
        P_obstacle_given_detect = bayes_update(P_obstacle, P_detect_if_obstacle, P_detect)
        
        print(f"\nResult: P(Obstacle|Detected) = {P_obstacle_given_detect:.1%}")
        print(f"Belief increased from {P_obstacle:.1%} to {P_obstacle_given_detect:.1%}")
        
    except NotImplementedError:
        print("\n⚠️  Bayes' rule not implemented yet!")
    
    print("\n💡 Tip: Start with the basic bayes_update() function,")
    print("   then build up to belief maps!")
