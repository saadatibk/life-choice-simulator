import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LifeDecisionSimulator:
    """
    AI-Powered Life Decision Simulator
    Predicts financial outcomes for major life decisions over a 10-year horizon
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.income_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.years = 10
        self.inflation_rate = 0.03  # 3% annual inflation
        
    def create_baseline_scenario(self, current_salary: float, age: int, 
                               experience_years: int, industry: str = "general") -> np.ndarray:
        """Create baseline income projection without any major changes"""
        
        # Industry multipliers for salary growth
        industry_multipliers = {
            "tech": 1.08,
            "finance": 1.06,
            "healthcare": 1.05,
            "education": 1.03,
            "general": 1.04
        }
        
        growth_rate = industry_multipliers.get(industry.lower(), 1.04)
        
        # Age-based career stage adjustments
        if age < 30:
            growth_rate *= 1.02  # Higher growth in early career
        elif age > 50:
            growth_rate *= 0.98  # Slower growth later in career
            
        baseline_income = []
        current_income = current_salary
        
        for year in range(self.years):
            # Apply growth rate with some randomness
            growth_factor = np.random.normal(growth_rate, 0.02)
            current_income *= growth_factor
            
            # Apply inflation
            inflated_income = current_income * (1 + self.inflation_rate) ** year
            baseline_income.append(inflated_income)
            
        return np.array(baseline_income)
    
    def simulate_education_scenario(self, current_salary: float, age: int,
                                  education_cost: float, education_years: int,
                                  expected_salary_increase: float = 0.3) -> Dict:
        """Simulate going back to school scenario"""
        
        scenario_income = []
        total_cost = education_cost
        
        # During education years: reduced/no income + costs
        for year in range(education_years):
            # Assume 20% income during school (part-time work)
            year_income = current_salary * 0.2 - (education_cost / education_years)
            scenario_income.append(year_income)
        
        # Post-education: higher salary
        new_salary = current_salary * (1 + expected_salary_increase)
        for year in range(education_years, self.years):
            # Apply growth to new higher salary
            growth_factor = 1.05 ** (year - education_years + 1)  # 5% annual growth
            inflated_salary = new_salary * growth_factor * (1 + self.inflation_rate) ** year
            scenario_income.append(inflated_salary)
        
        return {
            'income_projection': np.array(scenario_income),
            'total_investment': total_cost,
            'break_even_analysis': self._calculate_breakeven(scenario_income, total_cost)
        }
    
    def simulate_career_change_scenario(self, current_salary: float, age: int,
                                      new_field_salary_ratio: float = 0.8,
                                      transition_months: int = 6) -> Dict:
        """Simulate career change scenario"""
        
        scenario_income = []
        transition_cost = current_salary * (transition_months / 12) * 0.5  # Reduced income during transition
        
        # Transition period with reduced income
        transition_years = max(1, transition_months // 12)
        new_salary = current_salary * new_field_salary_ratio
        
        for year in range(self.years):
            if year < transition_years:
                # Transition year: mix of old and new salary
                year_income = current_salary * 0.3 + new_salary * 0.5
            else:
                # New career growth (often faster initially)
                growth_factor = 1.06 ** (year - transition_years + 1)
                year_income = new_salary * growth_factor
            
            inflated_income = year_income * (1 + self.inflation_rate) ** year
            scenario_income.append(inflated_income)
        
        return {
            'income_projection': np.array(scenario_income),
            'total_investment': transition_cost,
            'break_even_analysis': self._calculate_breakeven(scenario_income, transition_cost)
        }
    
    def simulate_family_scenario(self, current_salary: float, age: int,
                               num_children: int = 1, childcare_years: int = 5) -> Dict:
        """Simulate having children scenario"""
        
        scenario_income = []
        annual_child_cost = 15000 * num_children  # Average annual cost per child
        
        for year in range(self.years):
            base_income = current_salary * (1.04 ** year)  # Normal growth
            
            if year < childcare_years:
                # Reduced income due to time off, childcare costs
                year_income = base_income * 0.85 - annual_child_cost
            else:
                # Full income but still some child-related costs
                year_income = base_income - (annual_child_cost * 0.6)
            
            inflated_income = year_income * (1 + self.inflation_rate) ** year
            scenario_income.append(inflated_income)
        
        total_cost = annual_child_cost * self.years * 0.8  # Total estimated cost
        
        return {
            'income_projection': np.array(scenario_income),
            'total_investment': total_cost,
            'break_even_analysis': None  # Family decisions aren't purely financial
        }
    
    def _calculate_breakeven(self, scenario_income: List[float], 
                           total_investment: float) -> Optional[int]:
        """Calculate when the investment pays off"""
        cumulative_benefit = 0
        
        for year, income in enumerate(scenario_income):
            cumulative_benefit += income
            if cumulative_benefit >= total_investment:
                return year + 1
        
        return None  # Doesn't break even within timeframe
    
    def compare_scenarios(self, baseline: np.ndarray, scenario: Dict, 
                         scenario_name: str) -> Dict:
        """Compare baseline vs scenario outcomes"""
        
        scenario_income = scenario['income_projection']
        
        # Calculate cumulative values
        baseline_cumulative = np.cumsum(baseline)
        scenario_cumulative = np.cumsum(scenario_income)
        
        # Find crossover point
        crossover_year = None
        for year in range(len(baseline_cumulative)):
            if scenario_cumulative[year] > baseline_cumulative[year]:
                crossover_year = year + 1
                break
        
        # Calculate 10-year totals
        baseline_total = baseline_cumulative[-1]
        scenario_total = scenario_cumulative[-1]
        net_benefit = scenario_total - baseline_total
        
        # ROI calculation
        investment = scenario.get('total_investment', 0)
        roi = (net_benefit / investment * 100) if investment > 0 else 0
        
        return {
            'scenario_name': scenario_name,
            'baseline_10yr_total': baseline_total,
            'scenario_10yr_total': scenario_total,
            'net_benefit': net_benefit,
            'roi_percentage': roi,
            'crossover_year': crossover_year,
            'break_even_year': scenario.get('break_even_analysis'),
            'yearly_baseline': baseline,
            'yearly_scenario': scenario_income
        }
    
    def visualize_comparison(self, comparison: Dict):
        """Create visualization of the scenario comparison"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        years = range(1, self.years + 1)
        
        # 1. Annual Income Comparison
        ax1.plot(years, comparison['yearly_baseline'], 
                label='Baseline', marker='o', linewidth=2)
        ax1.plot(years, comparison['yearly_scenario'], 
                label=comparison['scenario_name'], marker='s', linewidth=2)
        ax1.set_title('Annual Income Projection')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Income ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Cumulative Income
        baseline_cum = np.cumsum(comparison['yearly_baseline'])
        scenario_cum = np.cumsum(comparison['yearly_scenario'])
        
        ax2.plot(years, baseline_cum, label='Baseline (Cumulative)', linewidth=3)
        ax2.plot(years, scenario_cum, label=f"{comparison['scenario_name']} (Cumulative)", linewidth=3)
        ax2.set_title('Cumulative Income Over Time')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Income ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 3. Net Benefit by Year
        net_benefit_yearly = comparison['yearly_scenario'] - comparison['yearly_baseline']
        colors = ['red' if x < 0 else 'green' for x in net_benefit_yearly]
        
        ax3.bar(years, net_benefit_yearly, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Annual Net Benefit/Loss')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Net Benefit ($)')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 4. Summary Statistics
        ax4.axis('off')
        summary_text = f"""
        SCENARIO ANALYSIS: {comparison['scenario_name']}
        
        10-Year Totals:
        • Baseline: ${comparison['baseline_10yr_total']:,.0f}
        • Scenario: ${comparison['scenario_10yr_total']:,.0f}
        
        Net Benefit: ${comparison['net_benefit']:,.0f}
        ROI: {comparison['roi_percentage']:.1f}%
        
        Timeline:
        • Break-even: Year {comparison['break_even_year'] or 'N/A'}
        • Crossover: Year {comparison['crossover_year'] or 'Never'}
        
        Recommendation:
        {'✅ Financially Favorable' if comparison['net_benefit'] > 0 else '⚠️  Consider Non-Financial Benefits'}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage and testing
def create_sample_data():
    """Create sample profiles for testing"""
    profiles = {
        'profile_1': {
            'name': 'Tech Professional - Returning to School',
            'current_salary': 85000,
            'age': 28,
            'experience_years': 5,
            'industry': 'tech'
        },
        'profile_2': {
            'name': 'Teacher - Career Change',
            'current_salary': 50000,
            'age': 35,
            'experience_years': 10,
            'industry': 'education'
        },
        'profile_3': {
            'name': 'Finance Worker - Starting Family',
            'current_salary': 95000,
            'age': 30,
            'experience_years': 7,
            'industry': 'finance'
        }
    }
    return profiles

if __name__ == "__main__":
    # Demo the simulator
    simulator = LifeDecisionSimulator()
    profiles = create_sample_data()
    
    # Example: Tech professional considering MBA
    profile = profiles['profile_1']
    print(f"Analyzing: {profile['name']}")
    
    # Generate baseline
    baseline = simulator.create_baseline_scenario(
        profile['current_salary'], 
        profile['age'], 
        profile['experience_years'], 
        profile['industry']
    )
    
    # Simulate MBA scenario
    mba_scenario = simulator.simulate_education_scenario(
        current_salary=profile['current_salary'],
        age=profile['age'],
        education_cost=120000,  # MBA cost
        education_years=2,
        expected_salary_increase=0.4  # 40% salary bump
    )
    
    # Compare scenarios
    comparison = simulator.compare_scenarios(baseline, mba_scenario, "MBA Program")
    
    # Print results
    print(f"Net 10-year benefit: ${comparison['net_benefit']:,.0f}")
    print(f"ROI: {comparison['roi_percentage']:.1f}%")
    print(f"Break-even year: {comparison['break_even_year']}")
