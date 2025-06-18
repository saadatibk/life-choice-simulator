# Life Decision Simulator - AI/ML Apprenticeship Project
# LinkedIn Reach Program

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Industry(Enum):
    """Enum for different industry types"""
    TECH = "tech"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    GENERAL = "general"
    CONSULTING = "consulting"
    MARKETING = "marketing"

@dataclass
class PersonProfile:
    """Data class for person profile"""
    name: str
    current_salary: float
    age: int
    experience_years: int
    industry: Industry
    education_level: str = "bachelor"
    location_cost_index: float = 1.0  # Cost of living multiplier
    risk_tolerance: str = "medium"  # low, medium, high

@dataclass
class ScenarioResult:
    """Data class for scenario results"""
    scenario_name: str
    income_projection: np.ndarray
    total_investment: float
    break_even_year: Optional[int]
    risk_score: float
    confidence_interval: Tuple[float, float]

class AdvancedLifeDecisionSimulator:
    """
    Enhanced AI-Powered Life Decision Simulator
    Features:
    - Multiple ML models for prediction
    - Monte Carlo simulations for uncertainty
    - Risk assessment
    - Interactive visualizations
    - Comprehensive scenario analysis
    """
    
    def __init__(self, simulation_years: int = 10, monte_carlo_runs: int = 1000):
        self.simulation_years = simulation_years
        self.monte_carlo_runs = monte_carlo_runs
        self.inflation_rate = 0.03
        
        # Initialize ML models
        self.income_models = {
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=150, random_state=42)
        }
        
        # Industry-specific parameters
        self.industry_data = {
            Industry.TECH: {'growth_rate': 1.08, 'volatility': 0.15, 'recession_impact': 0.85},
            Industry.FINANCE: {'growth_rate': 1.06, 'volatility': 0.12, 'recession_impact': 0.80},
            Industry.HEALTHCARE: {'growth_rate': 1.05, 'volatility': 0.08, 'recession_impact': 0.95},
            Industry.EDUCATION: {'growth_rate': 1.03, 'volatility': 0.05, 'recession_impact': 0.98},
            Industry.CONSULTING: {'growth_rate': 1.07, 'volatility': 0.18, 'recession_impact': 0.75},
            Industry.MARKETING: {'growth_rate': 1.05, 'volatility': 0.14, 'recession_impact': 0.82},
            Industry.GENERAL: {'growth_rate': 1.04, 'volatility': 0.10, 'recession_impact': 0.90}
        }
    
    def _generate_economic_scenarios(self) -> np.ndarray:
        """Generate realistic economic scenarios including recessions"""
        scenarios = []
        
        for _ in range(self.monte_carlo_runs):
            scenario = np.ones(self.simulation_years)
            
            # Random recession probability (10% chance each year)
            for year in range(self.simulation_years):
                if np.random.random() < 0.1:  # 10% recession chance
                    scenario[year] *= np.random.uniform(0.85, 0.95)  # 5-15% impact
                else:
                    scenario[year] *= np.random.normal(1.02, 0.03)  # Normal growth
            
            scenarios.append(scenario)
        
        return np.array(scenarios)
    
    def create_baseline_projection(self, profile: PersonProfile) -> ScenarioResult:
        """Create enhanced baseline projection with uncertainty"""
        industry_params = self.industry_data[profile.industry]
        
        # Monte Carlo simulation for baseline
        projections = []
        
        for _ in range(self.monte_carlo_runs):
            projection = []
            current_income = profile.current_salary
            
            for year in range(self.simulation_years):
                # Age-based career adjustments
                age_factor = self._calculate_age_factor(profile.age + year)
                
                # Industry growth with volatility
                growth_rate = industry_params['growth_rate']
                volatility = industry_params['volatility']
                annual_growth = np.random.normal(growth_rate, volatility * growth_rate)
                
                # Apply growth and age factor
                current_income *= annual_growth * age_factor
                
                # Apply inflation and location cost
                real_income = current_income * (1 + self.inflation_rate) ** year * profile.location_cost_index
                projection.append(real_income)
            
            projections.append(projection)
        
        projections = np.array(projections)
        mean_projection = np.mean(projections, axis=0)
        confidence_interval = (
            np.percentile(projections, 10, axis=0)[-1],
            np.percentile(projections, 90, axis=0)[-1]
        )
        
        return ScenarioResult(
            scenario_name="Baseline",
            income_projection=mean_projection,
            total_investment=0,
            break_even_year=None,
            risk_score=industry_params['volatility'],
            confidence_interval=confidence_interval
        )
    
    def _calculate_age_factor(self, age: int) -> float:
        """Calculate age-based career progression factor"""
        if age < 25:
            return 1.03  # High growth early career
        elif age < 35:
            return 1.02  # Good growth
        elif age < 45:
            return 1.01  # Moderate growth
        elif age < 55:
            return 1.005  # Slow growth
        else:
            return 0.998  # Potential decline
    
    def simulate_education_scenario(self, profile: PersonProfile, 
                                  education_cost: float, 
                                  education_years: int,
                                  expected_salary_multiplier: float = 1.4,
                                  education_type: str = "graduate") -> ScenarioResult:
        """Enhanced education scenario with realistic modeling"""
        
        projections = []
        
        for _ in range(self.monte_carlo_runs):
            projection = []
            total_cost = education_cost
            
            # During education phase
            for year in range(education_years):
                # Reduced income during education + costs
                year_income = profile.current_salary * 0.2 - (education_cost / education_years)
                projection.append(year_income)
            
            # Post-education enhanced salary
            new_base_salary = profile.current_salary * expected_salary_multiplier
            industry_params = self.industry_data[profile.industry]
            
            for year in range(education_years, self.simulation_years):
                # Enhanced growth rate post-education
                enhanced_growth = industry_params['growth_rate'] * 1.1  # 10% boost
                volatility = industry_params['volatility'] * 0.9  # Lower volatility with higher education
                
                growth_factor = np.random.normal(enhanced_growth, volatility * enhanced_growth)
                age_factor = self._calculate_age_factor(profile.age + year)
                
                year_income = new_base_salary * (growth_factor ** (year - education_years + 1)) * age_factor
                year_income *= (1 + self.inflation_rate) ** year * profile.location_cost_index
                
                projection.append(year_income)
            
            projections.append(projection)
        
        projections = np.array(projections)
        mean_projection = np.mean(projections, axis=0)
        
        # Calculate break-even
        cumulative_diff = np.cumsum(mean_projection) - total_cost
        break_even_year = None
        for year, cum_diff in enumerate(cumulative_diff):
            if cum_diff > 0:
                break_even_year = year + 1
                break
        
        # Risk assessment
        success_rate = np.mean([np.sum(proj) > total_cost for proj in projections])
        risk_score = 1 - success_rate
        
        confidence_interval = (
            np.percentile(projections, 10, axis=0)[-1],
            np.percentile(projections, 90, axis=0)[-1]
        )
        
        return ScenarioResult(
            scenario_name=f"{education_type.title()} Education",
            income_projection=mean_projection,
            total_investment=total_cost,
            break_even_year=break_even_year,
            risk_score=risk_score,
            confidence_interval=confidence_interval
        )
    
    def simulate_career_change_scenario(self, profile: PersonProfile,
                                      new_industry: Industry,
                                      salary_change_ratio: float = 0.8,
                                      transition_months: int = 6) -> ScenarioResult:
        """Enhanced career change scenario"""
        
        projections = []
        transition_cost = profile.current_salary * (transition_months / 12)
        
        for _ in range(self.monte_carlo_runs):
            projection = []
            new_salary = profile.current_salary * salary_change_ratio
            new_industry_params = self.industry_data[new_industry]
            
            for year in range(self.simulation_years):
                if year == 0:  # Transition year
                    year_income = new_salary * 0.7  # Reduced first year
                else:
                    # Accelerated growth in new field (catching up effect)
                    growth_rate = new_industry_params['growth_rate'] * 1.05
                    volatility = new_industry_params['volatility']
                    
                    growth_factor = np.random.normal(growth_rate ** year, volatility * growth_rate)
                    age_factor = self._calculate_age_factor(profile.age + year)
                    
                    year_income = new_salary * growth_factor * age_factor
                
                year_income *= (1 + self.inflation_rate) ** year * profile.location_cost_index
                projection.append(year_income)
            
            projections.append(projection)
        
        projections = np.array(projections)
        mean_projection = np.mean(projections, axis=0)
        
        # Risk assessment based on industry volatility and career change uncertainty
        base_risk = new_industry_params['volatility']
        transition_risk = 0.3  # Additional risk from career change
        combined_risk = min(base_risk + transition_risk, 1.0)
        
        confidence_interval = (
            np.percentile(projections, 15, axis=0)[-1],  # Wider interval due to higher uncertainty
            np.percentile(projections, 85, axis=0)[-1]
        )
        
        return ScenarioResult(
            scenario_name=f"Career Change to {new_industry.value.title()}",
            income_projection=mean_projection,
            total_investment=transition_cost,
            break_even_year=self._calculate_break_even(mean_projection, transition_cost),
            risk_score=combined_risk,
            confidence_interval=confidence_interval
        )
    
    def simulate_entrepreneurship_scenario(self, profile: PersonProfile,
                                         startup_cost: float,
                                         success_probability: float = 0.3,
                                         failure_income_ratio: float = 0.6,
                                         success_multiplier: float = 3.0) -> ScenarioResult:
        """Simulate entrepreneurship scenario with success/failure modeling"""
        
        projections = []
        
        for _ in range(self.monte_carlo_runs):
            projection = []
            
            # Determine if venture succeeds (based on success probability)
            venture_succeeds = np.random.random() < success_probability
            
            if venture_succeeds:
                # Success scenario - exponential growth
                base_income = profile.current_salary * success_multiplier
                for year in range(self.simulation_years):
                    # Exponential growth with high volatility
                    growth_factor = np.random.lognormal(0.15, 0.3)  # High variance
                    year_income = base_income * (growth_factor ** year)
                    year_income *= (1 + self.inflation_rate) ** year
                    projection.append(year_income)
            else:
                # Failure scenario - return to employment at lower salary
                recovery_salary = profile.current_salary * failure_income_ratio
                for year in range(self.simulation_years):
                    if year < 2:  # Recovery period
                        year_income = recovery_salary * (0.5 + 0.25 * year)
                    else:
                        # Normal growth after recovery
                        growth_factor = 1.04 ** (year - 1)
                        year_income = recovery_salary * growth_factor
                    
                    year_income *= (1 + self.inflation_rate) ** year
                    projection.append(year_income)
            
            projections.append(projection)
        
        projections = np.array(projections)
        mean_projection = np.mean(projections, axis=0)
        
        # High risk score due to uncertainty
        risk_score = 1 - success_probability
        
        confidence_interval = (
            np.percentile(projections, 5, axis=0)[-1],   # Very wide interval
            np.percentile(projections, 95, axis=0)[-1]
        )
        
        return ScenarioResult(
            scenario_name="Entrepreneurship",
            income_projection=mean_projection,
            total_investment=startup_cost,
            break_even_year=self._calculate_break_even(mean_projection, startup_cost),
            risk_score=risk_score,
            confidence_interval=confidence_interval
        )
    
    def _calculate_break_even(self, projection: np.ndarray, investment: float) -> Optional[int]:
        """Calculate break-even point"""
        cumulative = np.cumsum(projection)
        for year, cum_value in enumerate(cumulative):
            if cum_value >= investment:
                return year + 1
        return None
    
    def comprehensive_analysis(self, profile: PersonProfile, scenarios: List[ScenarioResult]) -> Dict:
        """Perform comprehensive analysis across all scenarios"""
        
        results = {}
        baseline = scenarios[0]  # Assume first scenario is baseline
        
        for scenario in scenarios[1:]:  # Skip baseline
            net_benefit = np.sum(scenario.income_projection) - np.sum(baseline.income_projection)
            roi = (net_benefit / scenario.total_investment * 100) if scenario.total_investment > 0 else 0
            
            # Risk-adjusted return
            risk_adjusted_return = roi / (1 + scenario.risk_score)
            
            # Sharpe-like ratio for life decisions
            sharpe_ratio = net_benefit / (scenario.total_investment * scenario.risk_score) if scenario.total_investment > 0 and scenario.risk_score > 0 else 0
            
            results[scenario.scenario_name] = {
                'net_benefit': net_benefit,
                'roi': roi,
                'risk_score': scenario.risk_score,
                'risk_adjusted_return': risk_adjusted_return,
                'sharpe_ratio': sharpe_ratio,
                'break_even_year': scenario.break_even_year,
                'confidence_interval': scenario.confidence_interval,
                'total_10yr_income': np.sum(scenario.income_projection)
            }
        
        # Rank scenarios by risk-adjusted return
        ranked_scenarios = sorted(results.items(), key=lambda x: x[1]['risk_adjusted_return'], reverse=True)
        
        return {
            'detailed_results': results,
            'ranked_scenarios': ranked_scenarios,
            'baseline_total': np.sum(baseline.income_projection)
        }
    
    def create_interactive_dashboard(self, profile: PersonProfile, scenarios: List[ScenarioResult], analysis: Dict):
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annual Income Projections', 'Cumulative Income', 
                          'Risk vs Return Analysis', 'Scenario Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        years = list(range(1, self.simulation_years + 1))
        colors = px.colors.qualitative.Set3
        
        # 1. Annual Income Projections
        for i, scenario in enumerate(scenarios):
            fig.add_trace(
                go.Scatter(
                    x=years, 
                    y=scenario.income_projection,
                    mode='lines+markers',
                    name=scenario.scenario_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    hovertemplate='Year %{x}<br>Income: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Cumulative Income
        for i, scenario in enumerate(scenarios):
            cumulative = np.cumsum(scenario.income_projection)
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=cumulative,
                    mode='lines',
                    name=f"{scenario.scenario_name} (Cumulative)",
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    hovertemplate='Year %{x}<br>Cumulative: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Risk vs Return Scatter
        if len(scenarios) > 1:
            scenario_names = [s.scenario_name for s in scenarios[1:]]  # Skip baseline
            returns = [analysis['detailed_results'][name]['roi'] for name in scenario_names]
            risks = [analysis['detailed_results'][name]['risk_score'] * 100 for name in scenario_names]
            
            fig.add_trace(
                go.Scatter(
                    x=risks,
                    y=returns,
                    mode='markers+text',
                    text=scenario_names,
                    textposition="top center",
                    marker=dict(size=15, color=colors[:len(scenario_names)]),
                    name='Risk vs Return',
                    hovertemplate='Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Summary Table
        if len(scenarios) > 1:
            table_data = []
            for name, data in analysis['detailed_results'].items():
                table_data.append([
                    name,
                    f"${data['net_benefit']:,.0f}",
                    f"{data['roi']:.1f}%",
                    f"{data['risk_score']*100:.1f}%",
                    f"Year {data['break_even_year']}" if data['break_even_year'] else "N/A"
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Scenario', 'Net Benefit', 'ROI', 'Risk', 'Break-even'],
                               fill_color='lightblue'),
                    cells=dict(values=list(zip(*table_data)),
                              fill_color='white')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title=f"Life Decision Analysis Dashboard - {profile.name}",
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Annual Income ($)", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Income ($)", row=1, col=2)
        fig.update_xaxes(title_text="Risk Score (%)", row=2, col=1)
        fig.update_yaxes(title_text="ROI (%)", row=2, col=1)
        
        return fig

# Helper function to create sample profiles
def create_sample_profiles() -> Dict[str, PersonProfile]:
    """Create diverse sample profiles for testing"""
    return {
        'tech_professional': PersonProfile(
            name="Sarah Chen - Software Engineer",
            current_salary=95000,
            age=28,
            experience_years=5,
            industry=Industry.TECH,
            education_level="bachelor",
            location_cost_index=1.3,  # High-cost area
            risk_tolerance="high"
        ),
        'teacher': PersonProfile(
            name="Michael Rodriguez - High School Teacher",
            current_salary=52000,
            age=32,
            experience_years=8,
            industry=Industry.EDUCATION,
            education_level="master",
            location_cost_index=0.9,  # Lower-cost area
            risk_tolerance="low"
        ),
        'finance_analyst': PersonProfile(
            name="Jessica Wang - Financial Analyst",
            current_salary=78000,
            age=26,
            experience_years=3,
            industry=Industry.FINANCE,
            education_level="bachelor",
            location_cost_index=1.2,
            risk_tolerance="medium"
        ),
        'consultant': PersonProfile(
            name="David Park - Management Consultant",
            current_salary=110000,
            age=29,
            experience_years=6,
            industry=Industry.CONSULTING,
            education_level="mba",
            location_cost_index=1.4,
            risk_tolerance="high"
        )
    }

# Main execution function
def run_comprehensive_analysis():
    """Run comprehensive analysis for all sample profiles"""
    
    print("🚀 Life Decision Simulator - Comprehensive Analysis")
    print("=" * 60)
    
    simulator = AdvancedLifeDecisionSimulator(simulation_years=10, monte_carlo_runs=500)
    profiles = create_sample_profiles()
    
    # Analysis for each profile
    for profile_key, profile in profiles.items():
        print(f"\n📊 Analyzing: {profile.name}")
        print("-" * 40)
        
        # Create scenarios
        baseline = simulator.create_baseline_projection(profile)
        
        # Education scenario (MBA/Graduate degree)
        education_scenario = simulator.simulate_education_scenario(
            profile, 
            education_cost=100000, 
            education_years=2, 
            expected_salary_multiplier=1.5,
            education_type="MBA"
        )
        
        # Career change scenario
        new_industry = Industry.TECH if profile.industry != Industry.TECH else Industry.CONSULTING
        career_change = simulator.simulate_career_change_scenario(
            profile, 
            new_industry=new_industry,
            salary_change_ratio=0.85,
            transition_months=8
        )
        
        # Entrepreneurship scenario
        entrepreneurship = simulator.simulate_entrepreneurship_scenario(
            profile,
            startup_cost=75000,
            success_probability=0.25,
            success_multiplier=4.0
        )
        
        scenarios = [baseline, education_scenario, career_change, entrepreneurship]
        
        # Comprehensive analysis
        analysis = simulator.comprehensive_analysis(profile, scenarios)
        
        # Print results
        print(f"Baseline 10-year total: ${analysis['baseline_total']:,.0f}")
        print("\nScenario Rankings (by Risk-Adjusted Return):")
        for i, (scenario_name, data) in enumerate(analysis['ranked_scenarios'], 1):
            print(f"{i}. {scenario_name}:")
            print(f"   • Net Benefit: ${data['net_benefit']:,.0f}")
            print(f"   • ROI: {data['roi']:.1f}%")
            print(f"   • Risk Score: {data['risk_score']*100:.1f}%")
            print(f"   • Risk-Adjusted Return: {data['risk_adjusted_return']:.1f}")
        
        # Create and display an interactive dashboard
        dashboard = simulator.create_interactive_dashboard(profile, scenarios, analysis)
        dashboard.show()
    
    print("\n✅ Analysis Complete!")
    print("Check the interactive dashboards above for detailed visualizations.")

# Run the analysis
if __name__ == "__main__":
    run_comprehensive_analysis()
