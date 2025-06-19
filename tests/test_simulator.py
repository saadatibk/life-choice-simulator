"""
Unit Tests for Life Choice Simulator

Run with: pytest tests/test_simulator.py
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from life_decision_simulator import *

class TestPersonProfile:
    """Test PersonProfile class"""
    
    def test_profile_creation(self):
        """Test basic profile creation"""
        profile = PersonProfile(
            name="Test User",
            current_salary=75000,
            age=30,
            experience_years=5,
            industry=Industry.TECH
        )
        
        assert profile.name == "Test User"
        assert profile.current_salary == 75000
        assert profile.age == 30
        assert profile.industry == Industry.TECH
        assert profile.education_level == "bachelor"  # default
        assert profile.location == Location.DENVER  # default
    
    def test_profile_with_all_fields(self):
        """Test profile with all fields specified"""
        skill_scores = {'technical': 0.8, 'leadership': 0.6}
        
        profile = PersonProfile(
            name="Full Test User",
            current_salary=100000,
            age=35,
            experience_years=10,
            industry=Industry.FINANCE,
            education_level="master",
            location=Location.SF_BAY_AREA,
            risk_tolerance="high",
            family_size=3,
            debt_amount=50000,
            savings=75000,
            skill_scores=skill_scores,
            career_satisfaction=8.5,
            work_life_balance=7.2
        )
        
        assert profile.education_level == "master"
        assert profile.risk_tolerance == "high"
        assert profile.family_size == 3
        assert profile.skill_scores == skill_scores

class TestEnhancedLifeDecisionSimulator:
    """Test main simulator class"""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance for testing"""
        add_optimize_method()  # Add missing method
        return EnhancedLifeDecisionSimulator(simulation_years=5, monte_carlo_runs=10)
    
    @pytest.fixture
    def test_profile(self):
        """Create test profile"""
        return PersonProfile(
            name="Test Profile",
            current_salary=80000,
            age=28,
            experience_years=6,
            industry=Industry.TECH,
            location=Location.DENVER,
            savings=30000,
            debt_amount=20000
        )
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator.simulation_years == 5
        assert simulator.monte_carlo_runs == 10
        assert simulator.inflation_rate == 0.028
        assert len(simulator.industry_data) == 10
    
    def test_age_factor_calculation(self, simulator):
        """Test age factor calculation"""
        assert simulator._calculate_age_factor(24) == 1.04  # Early career
        assert simulator._calculate_age_factor(28) == 1.03  # Strong growth
        assert simulator._calculate_age_factor(32) == 1.025  # Good growth
        assert simulator._calculate_age_factor(42) == 1.015  # Moderate growth
        assert simulator._calculate_age_factor(52) == 1.008  # Slow growth
        assert simulator._calculate_age_factor(62) == 1.002  # Minimal growth
        assert simulator._calculate_age_factor(67) == 0.995  # Potential decline
    
    def test_living_expenses_calculation(self, simulator, test_profile):
        """Test living expenses calculation"""
        expenses_year_0 = simulator._calculate_living_expenses(test_profile, 0)
        expenses_year_5 = simulator._calculate_living_expenses(test_profile, 5)
        
        # Expenses should increase over time due to inflation and age
        assert expenses_year_5 > expenses_year_0
        assert expenses_year_0 > 0
    
    def test_market_cycles_generation(self, simulator):
        """Test market cycles generation"""
        cycles = simulator._generate_market_cycles()
        
        assert cycles.shape == (simulator.monte_carlo_runs, simulator.simulation_years)
        assert np.all(cycles > 0)  # All cycles should be positive
        assert np.all(cycles < 2)  # Should be reasonable multipliers
    
    def test_baseline_projection(self, simulator, test_profile):
        """Test baseline projection creation"""
        baseline = simulator.create_enhanced_baseline_projection(test_profile)
        
        assert isinstance(baseline, ScenarioResult)
        assert baseline.scenario_name == "Baseline"
        assert len(baseline.income_projection) == simulator.simulation_years
        assert len(baseline.expense_projection) == simulator.simulation_years
        assert len(baseline.net_worth_projection) == simulator.simulation_years
        assert baseline.total_investment == 0
        assert 0 <= baseline.probability_of_success <= 1
        assert 1 <= baseline.stress_score <= 10
        assert 1 <= baseline.satisfaction_score <= 10
    
    def test_education_scenario(self, simulator, test_profile):
        """Test education scenario simulation"""
        education = simulator.simulate_advanced_education_scenario(
            test_profile,
            education_cost=50000,
            education_years=2,
            expected_salary_multiplier=1.4
        )
        
        assert isinstance(education, ScenarioResult)
        assert "Education" in education.scenario_name
        assert education.total_investment == 50000
        assert len(education.income_projection) == simulator.simulation_years
        assert 0 <= education.probability_of_success <= 1
    
    def test_stress_satisfaction_calculation(self, simulator, test_profile):
        """Test stress and satisfaction calculation"""
        stress, satisfaction = simulator._calculate_stress_satisfaction(
            "Test Scenario", test_profile, 0.2, 0.1
        )
        
        assert 1 <= stress <= 10
        assert 1 <= satisfaction <= 10

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_sample_profiles_creation(self):
        """Test sample profiles creation"""
        profiles = create_enhanced_sample_profiles()
        
        assert len(profiles) == 5
        assert 'tech_professional' in profiles
        assert 'teacher' in profiles
        assert 'finance_analyst' in profiles
        assert 'consultant' in profiles
        assert 'healthcare_worker' in profiles
        
        # Check profile types
        for profile in profiles.values():
            assert isinstance(profile, PersonProfile)
    
    def test_location_analysis(self):
        """Test location analysis function"""
        profiles = create_enhanced_sample_profiles()
        location_df = create_location_analysis(profiles)
        
        assert len(location_df) == len(profiles)
        assert 'Profile' in location_df.columns
        assert 'Location' in location_df.columns
        assert 'Purchasing_Power' in location_df.columns
    
    def test_personalized_recommendations(self):
        """Test recommendation generation"""
        profile = PersonProfile(
            name="Test",
            current_salary=75000,
            age=25,
            experience_years=3,
            industry=Industry.TECH,
            risk_tolerance="high",
            debt_amount=40000
        )
        
        # Create mock scenario
        baseline = ScenarioResult(
            scenario_name="Test",
            income_projection=np.array([75000] * 10),
            expense_projection=np.array([50000] * 10),
            net_worth_projection=np.array([10000] * 10),
            total_investment=0,
            break_even_year=None,
            risk_score=0.15,
            confidence_interval=(70000, 80000),
            probability_of_success=0.8,
            stress_score=5.0,
            satisfaction_score=7.0
        )
        
        recommendations = generate_personalized_recommendations(
            profile, [baseline], {}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

class TestIndustryData:
    """Test industry-specific data"""
    
    def test_industry_enum(self):
        """Test Industry enum"""
        assert Industry.TECH.value == "tech"
        assert Industry.FINANCE.value == "finance"
        assert len(Industry) == 10
    
    def test_location_enum(self):
        """Test Location enum"""
        sf_name, sf_cost = Location.SF_BAY_AREA.value
        assert sf_name == "San Francisco Bay Area"
        assert sf_cost > 1.0  # Should be expensive
        
        denver_name, denver_cost = Location.DENVER.value
        assert denver_name == "Denver"
        assert denver_cost == 1.0  # Baseline

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_age(self):
        """Test handling of edge case ages"""
        simulator = EnhancedLifeDecisionSimulator(simulation_years=5, monte_carlo_runs=10)
        add_optimize_method()
        
        # Very young age
        young_factor = simulator._calculate_age_factor(18)
        assert young_factor == 1.04
        
        # Very old age
        old_factor = simulator._calculate_age_factor(80)
        assert old_factor == 0.995
    
    def test_zero_debt_and_savings(self):
        """Test profile with zero debt and savings"""
        profile = PersonProfile(
            name="Zero Test",
            current_salary=60000,
            age=25,
            experience_years=2,
            industry=Industry.EDUCATION,
            debt_amount=0,
            savings=0
        )
        
        simulator = EnhancedLifeDecisionSimulator(simulation_years=3, monte_carlo_runs=5)
        add_optimize_method()
        
        baseline = simulator.create_enhanced_baseline_projection(profile)
        assert isinstance(baseline, ScenarioResult)

def test_quick_integration():
    """Integration test using the quick test function"""
    try:
        # This should run without errors
        from life_decision_simulator import run_quick_test
        run_quick_test()
        assert True  # If we get here, no exceptions were raised
    except Exception as e:
        pytest.fail(f"Quick test failed: {e}")

if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
