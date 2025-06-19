"""
Basic Usage Examples
Life Choice Simulator

This file demonstrates basic usage patterns for the Life Choice Simulator.
Run this after installing the package to see example outputs.
"""

import sys
import os

# Add parent directory to path to import the simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from life_decision_simulator import *
import numpy as np

def example_1_basic_profile():
    """Example 1: Create a basic user profile and run baseline analysis"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Profile Analysis")
    print("=" * 60)
    
    # Create a simple profile
    profile = PersonProfile(
        name="Alex Johnson - Software Developer",
        current_salary=85000,
        age=27,
        experience_years=5,
        industry=Industry.TECH,
        location=Location.AUSTIN,
        risk_tolerance="medium",
        family_size=1,
        debt_amount=30000,
        savings=40000,
        skill_scores={'technical': 0.8, 'leadership': 0.5, 'communication': 0.6}
    )
    
    # Initialize simulator
    simulator = EnhancedLifeDecisionSimulator(simulation_years=10, monte_carlo_runs=500)
    add_optimize_method()  # Add missing method
    
    # Run baseline analysis
    baseline = simulator.create_enhanced_baseline_projection(profile)
    
    print(f"Profile: {profile.name}")
    print(f"Current Salary: ${profile.current_salary:,}")
    print(f"Location: {profile.location.value[0]}")
    print(f"\nBaseline Results:")
    print(f"10-year total income: ${np.sum(baseline.income_projection):,.0f}")
    print(f"Final net worth: ${baseline.net_worth_projection[-1]:,.0f}")
    print(f"Success probability: {baseline.probability_of_success*100:.1f}%")
    print(f"Stress score: {baseline.stress_score:.1f}/10")
    print(f"Satisfaction score: {baseline.satisfaction_score:.1f}/10")

def example_2_education_scenario():
    """Example 2: Compare baseline vs education investment"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Education Investment Analysis")
    print("=" * 60)
    
    # Create profile for someone considering graduate school
    profile = PersonProfile(
        name="Maria Garcia - Marketing Analyst",
        current_salary=65000,
        age=26,
        experience_years=3,
        industry=Industry.MARKETING,
        location=Location.CHICAGO,
        risk_tolerance="medium",
        family_size=1,
        debt_amount=25000,
        savings=35000
    )
    
    simulator = EnhancedLifeDecisionSimulator(simulation_years=15, monte_carlo_runs=500)
    add_optimize_method()
    
    # Baseline scenario
    baseline = simulator.create_enhanced_baseline_projection(profile)
    
    # MBA scenario
    mba_scenario = simulator.simulate_advanced_education_scenario(
        profile,
        education_cost=90000,
        education_years=2,
        expected_salary_multiplier=1.6,
        education_type="MBA"
    )
    
    print(f"Profile: {profile.name}")
    print(f"Current Salary: ${profile.current_salary:,}")
    
    print(f"\nBaseline (No Education):")
    print(f"15-year total: ${np.sum(baseline.income_projection):,.0f}")
    print(f"Final net worth: ${baseline.net_worth_projection[-1]:,.0f}")
    
    print(f"\nMBA Scenario:")
    print(f"Investment: ${mba_scenario.total_investment:,}")
    print(f"15-year total: ${np.sum(mba_scenario.income_projection):,.0f}")
    print(f"Final net worth: ${mba_scenario.net_worth_projection[-1]:,.0f}")
    print(f"Break-even: {mba_scenario.break_even_year} years")
    
    # Calculate ROI
    net_benefit = np.sum(mba_scenario.income_projection) - np.sum(baseline.income_projection)
    roi = (net_benefit / mba_scenario.total_investment) * 100
    print(f"ROI: {roi:.1f}%")

def example_3_location_comparison():
    """Example 3: Compare different locations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Location Comparison Analysis")
    print("=" * 60)
    
    base_profile = PersonProfile(
        name="David Kim - Consultant",
        current_salary=120000,
        age=30,
        experience_years=7,
        industry=Industry.CONSULTING,
        risk_tolerance="high",
        family_size=2,
        debt_amount=50000,
        savings=80000
    )
    
    simulator = EnhancedLifeDecisionSimulator(simulation_years=10, monte_carlo_runs=300)
    add_optimize_method()
    
    locations_to_compare = [
        Location.SF_BAY_AREA,
        Location.NYC,
        Location.AUSTIN,
        Location.DENVER
    ]
    
    print(f"Comparing locations for: {base_profile.name}")
    print(f"Base salary: ${base_profile.current_salary:,}")
    print("\nLocation Analysis:")
    print("-" * 40)
    
    for location in locations_to_compare:
        # Create profile copy with different location
        profile = PersonProfile(
            name=base_profile.name,
            current_salary=base_profile.current_salary,
            age=base_profile.age,
            experience_years=base_profile.experience_years,
            industry=base_profile.industry,
            location=location,
            risk_tolerance=base_profile.risk_tolerance,
            family_size=base_profile.family_size,
            debt_amount=base_profile.debt_amount,
            savings=base_profile.savings
        )
        
        scenario = simulator.create_enhanced_baseline_projection(profile)
        location_name, cost_index = location.value
        purchasing_power = base_profile.current_salary / cost_index
        
        print(f"{location_name}:")
        print(f"  Cost index: {cost_index:.2f}")
        print(f"  Purchasing power: ${purchasing_power:,.0f}")
        print(f"  10-year total: ${np.sum(scenario.income_projection):,.0f}")
        print(f"  Final net worth: ${scenario.net_worth_projection[-1]:,.0f}")
        print()

def example_4_risk_analysis():
    """Example 4: Risk tolerance impact"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Risk Tolerance Analysis")
    print("=" * 60)
    
    base_profile = PersonProfile(
        name="Jennifer Wu - Finance Analyst",
        current_salary=78000,
        age=28,
        experience_years=5,
        industry=Industry.FINANCE,
        location=Location.NYC,
        family_size=1,
        debt_amount=20000,
        savings=45000
    )
    
    simulator = EnhancedLifeDecisionSimulator(simulation_years=12, monte_carlo_runs=400)
    add_optimize_method()
    
    risk_levels = ["low", "medium", "high"]
    
    print(f"Risk analysis for: {base_profile.name}")
    print("\nRisk Level Impact:")
    print("-" * 30)
    
    for risk_level in risk_levels:
        profile = PersonProfile(
            name=base_profile.name,
            current_salary=base_profile.current_salary,
            age=base_profile.age,
            experience_years=base_profile.experience_years,
            industry=base_profile.industry,
            location=base_profile.location,
            risk_tolerance=risk_level,
            family_size=base_profile.family_size,
            debt_amount=base_profile.debt_amount,
            savings=base_profile.savings
        )
        
        scenario = simulator.create_enhanced_baseline_projection(profile)
        recommendations = generate_personalized_recommendations(profile, [scenario], {})
        
        print(f"{risk_level.upper()} Risk Tolerance:")
        print(f"  Stress score: {scenario.stress_score:.1f}/10")
        print(f"  Satisfaction: {scenario.satisfaction_score:.1f}/10")
        print(f"  Success rate: {scenario.probability_of_success*100:.1f}%")
        print(f"  Key recommendation: {recommendations[0] if recommendations else 'N/A'}")
        print()

def main():
    """Run all examples"""
    print("🚀 Enhanced Life Decision Simulator - Basic Usage Examples")
    print("LinkedIn Reach Program Project")
    
    try:
        example_1_basic_profile()
        example_2_education_scenario()
        example_3_location_comparison()
        example_4_risk_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
