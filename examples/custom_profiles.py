"""
This module provides customizable profile templates and builders for different
life situations, career stages, and demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
import json

class LifeStage(Enum):
    """Life stage classifications"""
    EARLY_CAREER = "early_career"
    MID_CAREER = "mid_career"
    SENIOR_CAREER = "senior_career"
    PRE_RETIREMENT = "pre_retirement"
    RETIREMENT = "retirement"

class RiskTolerance(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

class IncomeLevel(Enum):
    """Income level classifications"""
    LOW = "low"           # <$50k
    MIDDLE = "middle"     # $50k-$100k
    UPPER_MIDDLE = "upper_middle"  # $100k-$200k
    HIGH = "high"         # $200k-$500k
    VERY_HIGH = "very_high"        # >$500k

@dataclass
class FinancialGoals:
    """Financial goals and priorities"""
    retirement_target_age: int = 65
    retirement_income_replacement: float = 0.8  # 80% of current income
    emergency_fund_months: int = 6
    debt_payoff_priority: List[str] = field(default_factory=lambda: ["high_interest", "student_loans", "mortgage"])
    wealth_building_priority: List[str] = field(default_factory=lambda: ["retirement", "investments", "real_estate"])
    legacy_goals: float = 0  # Desired inheritance amount
    financial_independence_target: float = 0  # FI target amount

@dataclass
class CareerGoals:
    """Career development goals"""
    target_roles: List[str] = field(default_factory=list)
    skill_development_priorities: List[str] = field(default_factory=list)
    industry_interests: List[str] = field(default_factory=list)
    leadership_aspirations: bool = False
    entrepreneurship_interest: float = 0.0  # 0-1 scale
    work_life_balance_priority: float = 0.7  # 0-1 scale
    geographic_flexibility: float = 0.5  # 0-1 scale
    remote_work_preference: float = 0.5  # 0-1 scale

@dataclass
class PersonalValues:
    """Personal values and priorities"""
    family_importance: float = 0.8  # 0-1 scale
    career_importance: float = 0.7
    health_importance: float = 0.9
    financial_security_importance: float = 0.8
    adventure_importance: float = 0.4
    stability_importance: float = 0.7
    social_impact_importance: float = 0.5
    personal_growth_importance: float = 0.8

@dataclass
class LifestylePreferences:
    """Lifestyle and spending preferences"""
    housing_preference: str = "own"  # "own", "rent", "flexible"
    location_preference: str = "suburban"  # "urban", "suburban", "rural"
    travel_spending_priority: float = 0.6  # 0-1 scale
    luxury_spending_tolerance: float = 0.4
    environmental_consciousness: float = 0.6
    social_spending_priority: float = 0.5
    health_wellness_spending: float = 0.7

@dataclass
class ComprehensiveProfile:
    """Comprehensive user profile for life decision analysis"""
    
    # Basic Demographics
    name: str = "User"
    age: int = 30
    birth_date: Optional[date] = None
    gender: str = "prefer_not_to_say"
    education_level: str = "bachelor"
    location: str = "us_national"
    
    # Family & Relationships
    marital_status: str = "single"  # single, married, divorced, widowed
    family_size: int = 1
    dependents: List[Dict[str, Any]] = field(default_factory=list)
    partner_profile: Optional['ComprehensiveProfile'] = None
    
    # Financial Information
    current_salary: float = 75000
    other_income: Dict[str, float] = field(default_factory=dict)
    savings: float = 25000
    checking_balance: float = 5000
    investments: Dict[str, float] = field(default_factory=dict)
    retirement_accounts: Dict[str, float] = field(default_factory=dict)
    debts: Dict[str, float] = field(default_factory=dict)
    monthly_expenses: Dict[str, float] = field(default_factory=dict)
    
    # Career Information
    current_job_title: str = "Professional"
    industry: str = "technology"
    years_experience: int = 5
    career_satisfaction: float = 7.0  # 1-10 scale
    job_security: float = 7.0
    advancement_potential: float = 6.0
    skills: Dict[str, float] = field(default_factory=dict)  # skill: proficiency (0-1)
    
    # Health & Lifestyle
    health_score: float = 8.0  # 1-10 scale
    fitness_level: float = 6.0
    stress_level: float = 5.0
    work_life_balance: float = 7.0
    life_satisfaction: float = 7.5
    
    # Goals and Preferences
    financial_goals: FinancialGoals = field(default_factory=FinancialGoals)
    career_goals: CareerGoals = field(default_factory=CareerGoals)
    personal_values: PersonalValues = field(default_factory=PersonalValues)
    lifestyle_preferences: LifestylePreferences = field(default_factory=LifestylePreferences)
    
    # Classification Attributes
    life_stage: LifeStage = LifeStage.EARLY_CAREER
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    income_level: IncomeLevel = IncomeLevel.MIDDLE
    
    def __post_init__(self):
        """Auto-classify profile attributes"""
        self.life_stage = self._determine_life_stage()
        self.income_level = self._determine_income_level()
        if not self.birth_date and self.age:
            current_year = datetime.now().year
            self.birth_date = date(current_year - self.age, 1, 1)
    
    def _determine_life_stage(self) -> LifeStage:
        """Determine life stage based on age and career factors"""
        if self.age < 30:
            return LifeStage.EARLY_CAREER
        elif self.age < 45:
            return LifeStage.MID_CAREER
        elif self.age < 55:
            return LifeStage.SENIOR_CAREER
        elif self.age < 65:
            return LifeStage.PRE_RETIREMENT
        else:
            return LifeStage.RETIREMENT
    
    def _determine_income_level(self) -> IncomeLevel:
        """Determine income level classification"""
        total_income = self.current_salary + sum(self.other_income.values())
        
        if total_income < 50000:
            return IncomeLevel.LOW
        elif total_income < 100000:
            return IncomeLevel.MIDDLE
        elif total_income < 200000:
            return IncomeLevel.UPPER_MIDDLE
        elif total_income < 500000:
            return IncomeLevel.HIGH
        else:
            return IncomeLevel.VERY_HIGH
    
    def calculate_net_worth(self) -> float:
        """Calculate total net worth"""
        assets = (self.savings + self.checking_balance + 
                 sum(self.investments.values()) + 
                 sum(self.retirement_accounts.values()))
        liabilities = sum(self.debts.values())
        return assets - liabilities
    
    def calculate_monthly_cash_flow(self) -> float:
        """Calculate monthly cash flow"""
        monthly_income = (self.current_salary + sum(self.other_income.values())) / 12
        monthly_expenses_total = sum(self.monthly_expenses.values())
        return monthly_income - monthly_expenses_total
    
    def get_financial_health_score(self) -> Dict[str, float]:
        """Calculate comprehensive financial health metrics"""
        net_worth = self.calculate_net_worth()
        monthly_cash_flow = self.calculate_monthly_cash_flow()
        monthly_income = (self.current_salary + sum(self.other_income.values())) / 12
        
        # Emergency fund ratio
        emergency_fund_ratio = self.savings / (sum(self.monthly_expenses.values()) * 6) if self.monthly_expenses else 1
        
        # Debt-to-income ratio
        total_debt = sum(self.debts.values())
        debt_to_income = (total_debt / 12) / monthly_income if monthly_income > 0 else 0
        
        # Savings rate
        savings_rate = monthly_cash_flow / monthly_income if monthly_income > 0 else 0
        
        # Investment diversification (simplified)
        investment_accounts = len([v for v in self.investments.values() if v > 0])
        diversification_score = min(1.0, investment_accounts / 5.0)  # Assume 5 is well-diversified
        
        return {
            'emergency_fund_ratio': min(1.0, emergency_fund_ratio),
            'debt_to_income': debt_to_income,
            'savings_rate': savings_rate,
            'diversification_score': diversification_score,
            'overall_score': np.mean([
                min(1.0, emergency_fund_ratio),
                max(0, 1 - debt_to_income),
                max(0, savings_rate),
                diversification_score
            ])
        }

class ProfileBuilder:
    """Builder class for creating customized profiles"""
    
    def __init__(self):
        self.profile = ComprehensiveProfile()
        self.load_templates()
    
    def load_templates(self):
        """Load profile templates for different scenarios"""
        self.templates = {
            'recent_graduate': {
                'age': 22,
                'years_experience': 0,
                'current_salary': 45000,
                'savings': 5000,
                'debts': {'student_loans': 25000},
                'life_stage': LifeStage.EARLY_CAREER,
                'career_goals': CareerGoals(
                    skill_development_priorities=['technical_skills', 'communication', 'leadership'],
                    leadership_aspirations=False,
                    work_life_balance_priority=0.8
                )
            },
            'young_professional': {
                'age': 28,
                'years_experience': 5,
                'current_salary': 75000,
                'savings': 35000,
                'investments': {'401k': 25000, 'index_funds': 15000},
                'debts': {'student_loans': 15000},
                'career_satisfaction': 7.0
            },
            'mid_career_family': {
                'age': 38,
                'marital_status': 'married',
                'family_size': 4,
                'current_salary': 125000,
                'savings': 50000,
                'investments': {'401k': 150000, 'ira': 75000, 'brokerage': 50000},
                'debts': {'mortgage': 300000},
                'dependents': [
                    {'type': 'child', 'age': 8, 'education_costs': 15000},
                    {'type': 'child', 'age': 12, 'education_costs': 18000}
                ]
            },
            'senior_executive': {
                'age': 48,
                'current_salary': 250000,
                'other_income': {'bonus': 75000, 'stock_options': 50000},
                'savings': 100000,
                'investments': {'401k': 400000, 'ira': 150000, 'brokerage': 300000, 'real_estate': 200000},
                'career_satisfaction': 8.5,
                'leadership_aspirations': True
            },
            'pre_retiree': {
                'age': 58,
                'current_salary': 180000,
                'savings': 150000,
                'investments': {'401k': 800000, 'ira': 300000, 'brokerage': 400000},
                'financial_goals': FinancialGoals(
                    retirement_target_age=62,
                    retirement_income_replacement=0.8,
                    legacy_goals=500000
                )
            },
            'entrepreneur': {
                'age': 32,
                'current_salary': 0,
                'other_income': {'business_income': 85000},
                'savings': 75000,
                'investments': {'business_equity': 200000, 'ira': 45000},
                'career_goals': CareerGoals(
                    entrepreneurship_interest=1.0,
                    geographic_flexibility=0.9,
                    remote_work_preference=0.9
                ),
                'risk_tolerance': RiskTolerance.AGGRESSIVE
            },
            'freelancer': {
                'age': 29,
                'other_income': {'freelance_income': 65000},
                'savings': 25000,
                'monthly_expenses': {'variable_income_buffer': 2000},
                'career_goals': CareerGoals(
                    work_life_balance_priority=0.9,
                    geographic_flexibility=1.0,
                    remote_work_preference=1.0
                )
            },
            'career_changer': {
                'age': 35,
                'current_salary': 95000,
                'savings': 60000,
                'career_satisfaction': 4.0,
                'career_goals': CareerGoals(
                    target_roles=['product_manager', 'consultant', 'entrepreneur'],
                    skill_development_priorities=['business_strategy', 'leadership', 'industry_knowledge']
                )
            }
        }
    
    def from_template(self, template_name: str) -> 'ProfileBuilder':
        """Start with a predefined template"""
        if template_name in self.templates:
            template_data = self.templates[template_name]
            
            # Apply template data to profile
            for key, value in template_data.items():
                if hasattr(self.profile, key):
                    if isinstance(value, dict) and hasattr(getattr(self.profile, key), '__dict__'):
                        # Handle nested objects like career_goals
                        for nested_key, nested_value in value.items():
                            setattr(getattr(self.profile, key), nested_key, nested_value)
                    else:
                        setattr(self.profile, key, value)
        
        return self
    
    def with_demographics(self, age: int, name: str = None, location: str = None, 
                         education_level: str = None) -> 'ProfileBuilder':
        """Set demographic information"""
        self.profile.age = age
        if name:
            self.profile.name = name
        if location:
            self.profile.location = location
        if education_level:
            self.profile.education_level = education_level
        return self
    
    def with_family(self, marital_status: str, family_size: int = None, 
                   dependents: List[Dict] = None) -> 'ProfileBuilder':
        """Set family information"""
        self.profile.marital_status = marital_status
        if family_size:
            self.profile.family_size = family_size
        if dependents:
            self.profile.dependents = dependents
        return self
    
    def with_finances(self, salary: float, savings: float = None, 
                     investments: Dict[str, float] = None, 
                     debts: Dict[str, float] = None) -> 'ProfileBuilder':
        """Set financial information"""
        self.profile.current_salary = salary
        if savings:
            self.profile.savings = savings
        if investments:
            self.profile.investments.update(investments)
        if debts:
            self.profile.debts.update(debts)
        return self
    
    def with_career(self, job_title: str, industry: str = None, 
                   years_experience: int = None, satisfaction: float = None) -> 'ProfileBuilder':
        """Set career information"""
        self.profile.current_job_title = job_title
        if industry:
            self.profile.industry = industry
        if years_experience:
            self.profile.years_experience = years_experience
        if satisfaction:
            self.profile.career_satisfaction = satisfaction
        return self
    
    def with_goals(self, financial_goals: FinancialGoals = None, 
                  career_goals: CareerGoals = None) -> 'ProfileBuilder':
        """Set goals and aspirations"""
        if financial_goals:
            self.profile.financial_goals = financial_goals
        if career_goals:
            self.profile.career_goals = career_goals
        return self
    
    def with_risk_tolerance(self, risk_level: RiskTolerance) -> 'ProfileBuilder':
        """Set risk tolerance"""
        self.profile.risk_tolerance = risk_level
        return self
    
    def with_monthly_expenses(self, expenses: Dict[str, float]) -> 'ProfileBuilder':
        """Set monthly expense breakdown"""
        self.profile.monthly_expenses.update(expenses)
        return self
    
    def with_skills(self, skills: Dict[str, float]) -> 'ProfileBuilder':
        """Set skill levels"""
        self.profile.skills.update(skills)
        return self
    
    def build(self) -> ComprehensiveProfile:
        """Build and return the completed profile"""
        # Auto-calculate derived fields
        self.profile.__post_init__()
        return self.profile

class ProfileValidator:
    """Validate and provide recommendations for profile completeness"""
    
    @staticmethod
    def validate_profile(profile: ComprehensiveProfile) -> Dict[str, Any]:
        """Validate profile and return completeness assessment"""
        validation_results = {
            'is_valid': True,
            'completeness_score': 0.0,
            'missing_fields': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check required fields
        required_fields = ['age', 'current_salary', 'savings']
        missing_required = []
        
        for field in required_fields:
            if not hasattr(profile, field) or getattr(profile, field) is None:
                missing_required.append(field)
        
        if missing_required:
            validation_results['is_valid'] = False
            validation_results['missing_fields'] = missing_required
        
        # Calculate completeness score
        all_fields = [
            'age', 'current_salary', 'savings', 'investments', 'debts',
            'career_satisfaction', 'health_score', 'monthly_expenses',
            'skills', 'financial_goals', 'career_goals'
        ]
        
        completed_fields = 0
        for field in all_fields:
            if hasattr(profile, field):
                value = getattr(profile, field)
                if value is not None and value != {} and value != []:
                    completed_fields += 1
        
        validation_results['completeness_score'] = completed_fields / len(all_fields)
        
        # Generate warnings and recommendations
        warnings, recommendations = ProfileValidator._generate_recommendations(profile)
        validation_results['warnings'] = warnings
        validation_results['recommendations'] = recommendations
        
        return validation_results
    
    @staticmethod
    def _generate_recommendations(profile: ComprehensiveProfile) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations for profile improvement"""
        warnings = []
        recommendations = []
        
        # Financial health checks
        financial_health = profile.get_financial_health_score()
        
        if financial_health['emergency_fund_ratio'] < 0.5:
            warnings.append("Emergency fund below recommended 6-month expenses")
            recommendations.append("Build emergency fund to cover 6 months of expenses")
        
        if financial_health['debt_to_income'] > 0.3:
            warnings.append("High debt-to-income ratio")
            recommendations.append("Focus on debt reduction strategies")
        
        if financial_health['savings_rate'] < 0.1:
            warnings.append("Low savings rate")
            recommendations.append("Increase monthly savings to at least 10% of income")
        
        # Career satisfaction checks
        if profile.career_satisfaction < 5.0:
            warnings.append("Low career satisfaction score")
            recommendations.append("Consider career development or transition planning")
        
        # Life stage specific recommendations
        if profile.life_stage == LifeStage.EARLY_CAREER:
            if not profile.skills:
                recommendations.append("Define and develop key professional skills")
            if sum(profile.retirement_accounts.values()) < profile.current_salary * 0.1:
                recommendations.append("Start retirement savings early for compound growth")
        
        elif profile.life_stage == LifeStage.MID_CAREER:
            if profile.family_size > 1 and not profile.dependents:
                recommendations.append("Plan for family education and care costs")
            if sum(profile.investments.values()) < profile.current_salary * 2:
                recommendations.append("Accelerate wealth building during peak earning years")
        
        elif profile.life_stage == LifeStage.PRE_RETIREMENT:
            retirement_assets = sum(profile.retirement_accounts.values()) + sum(profile.investments.values())
            if retirement_assets < profile.current_salary * 10:
                warnings.append("Retirement savings may be insufficient")
                recommendations.append("Consider catch-up contributions and delayed retirement")
        
        return warnings, recommendations

def create_sample_profiles() -> Dict[str, ComprehensiveProfile]:
    """Create sample profiles for different scenarios"""
    builder = ProfileBuilder()
    
    profiles = {}
    
    # Recent Graduate Profile
    profiles['recent_grad'] = (builder
        .from_template('recent_graduate')
        .with_demographics(23, "Alex Chen", "san_francisco")
        .with_skills({
            'python': 0.7,
            'data_analysis': 0.6,
            'communication': 0.5,
            'project_management': 0.3
        })
        .build())
    
    # Young Professional Profile
    builder = ProfileBuilder()
    profiles['young_professional'] = (builder
        .from_template('young_professional')
        .with_demographics(28, "Sarah Johnson", "austin")
        .with_monthly_expenses({
            'rent': 1800,
            'food': 600,
            'transportation': 400,
            'entertainment': 300,
            'utilities': 150,
            'insurance': 200,
            'other': 250
        })
        .build())
    
    # Family-Focused Profile
    builder = ProfileBuilder()
    profiles['family_focused'] = (builder
        .from_template('mid_career_family')
        .with_demographics(38, "Michael Rodriguez", "denver")
        .with_goals(
            financial_goals=FinancialGoals(
                retirement_target_age=60,
                emergency_fund_months=8,
                legacy_goals=200000
            ),
            career_goals=CareerGoals(
                work_life_balance_priority=0.9,
                leadership_aspirations=True,
                geographic_flexibility=0.3
            )
        )
        .build())
    
    # High Earner Profile
    builder = ProfileBuilder()
    profiles['high_earner'] = (builder
        .from_template('senior_executive')
        .with_demographics(45, "Jennifer Kim", "new_york")
        .with_risk_tolerance(RiskTolerance.AGGRESSIVE)
        .with_monthly_expenses({
            'mortgage': 4500,
            'food': 1200,
            'transportation': 800,
            'childcare': 2000,
            'entertainment': 1000,
            'utilities': 300,
            'insurance': 500,
            'other': 700
        })
        .build())
    
    # Entrepreneur Profile
    builder = ProfileBuilder()
    profiles['entrepreneur'] = (builder
        .from_template('entrepreneur')
        .with_demographics(32, "David Park", "remote")
        .with_finances(
            salary=0,
            savings=75000,
            investments={
                'business_equity': 200000,
                'ira': 45000,
                'emergency_fund': 30000
            }
        )
        .build())
    
    return profiles

def analyze_profile_scenarios(profiles: Dict[str, ComprehensiveProfile]) -> Dict[str, Any]:
    """Analyze different profile scenarios and their characteristics"""
    analysis = {}
    
    for name, profile in profiles.items():
        # Basic metrics
        net_worth = profile.calculate_net_worth()
        cash_flow = profile.calculate_monthly_cash_flow()
        financial_health = profile.get_financial_health_score()
        
        # Risk assessment
        risk_factors = []
        if financial_health['emergency_fund_ratio'] < 0.5:
            risk_factors.append('insufficient_emergency_fund')
        if financial_health['debt_to_income'] > 0.3:
            risk_factors.append('high_debt_burden')
        if len(profile.investments) < 2:
            risk_factors.append('low_diversification')
        
        # Opportunities
        opportunities = []
        if profile.age < 30 and sum(profile.retirement_accounts.values()) < 50000:
            opportunities.append('early_retirement_savings')
        if profile.career_satisfaction < 6 and profile.savings > profile.current_salary * 0.5:
            opportunities.append('career_transition_opportunity')
        if financial_health['savings_rate'] > 0.2:
            opportunities.append('accelerated_wealth_building')
        
        analysis[name] = {
            'profile_summary': {
                'name': profile.name,
                'age': profile.age,
                'life_stage': profile.life_stage.value,
                'income_level': profile.income_level.value,
                'risk_tolerance': profile.risk_tolerance.value
            },
            'financial_metrics': {
                'net_worth': net_worth,
                'monthly_cash_flow': cash_flow,
                'financial_health_score': financial_health['overall_score'],
                'savings_rate': financial_health['savings_rate']
            },
            'risk_factors': risk_factors,
            'opportunities': opportunities,
            'validation': ProfileValidator.validate_profile(profile)
        }
    
    return analysis

# Example usage and demonstration
if __name__ == "__main__":
    print("🏗️ Custom Profiles Demo - Enhanced Life Decision Simulator")
    print("=" * 65)
    
    # Create sample profiles
    sample_profiles = create_sample_profiles()
    
    print(f"\n📊 Created {len(sample_profiles)} sample profiles:")
    for name, profile in sample_profiles.items():
        print(f"• {name}: {profile.name} ({profile.age}), {profile.life_stage.value}")
    
    # Analyze profiles
    print("\n🔍 Profile Analysis:")
    print("-" * 40)
    
    analysis = analyze_profile_scenarios(sample_profiles)
    
    for name, data in analysis.items():
        print(f"\n{name.title().replace('_', ' ')}:")
        summary = data['profile_summary']
        metrics = data['financial_metrics']
        
        print(f"  Age: {summary['age']}, Stage: {summary['life_stage']}")
        print(f"  Net Worth: ${metrics['net_worth']:,.0f}")
        print(f"  Monthly Cash Flow: ${metrics['monthly_cash_flow']:,.0f}")
        print(f"  Financial Health: {metrics['financial_health_score']:.1%}")
        print(f"  Savings Rate: {metrics['savings_rate']:.1%}")
        
        if data['risk_factors']:
            print(f"  ⚠️ Risk Factors: {', '.join(data['risk_factors'])}")
        
        if data['opportunities']:
            print(f"  🎯 Opportunities: {', '.join(data['opportunities'])}")
        
        validation = data['validation']
        print(f"  📋 Profile Completeness: {validation['completeness_score']:.1%}")
    
    # Demonstrate profile builder
    print("\n🏗️ Custom Profile Builder Demo:")
    print("-" * 40)
    
    custom_profile = (ProfileBuilder()
        .with_demographics(35, "Custom User", "chicago")
        .with_family("married", family_size=3)
        .with_finances(
            salary=95000,
            savings=45000,
            investments={'401k': 85000, 'ira': 25000, 'stocks': 15000},
            debts={'mortgage': 280000, 'car_loan': 12000}
        )
        .with_career("Software Engineer", "technology", 10, 7.5)
        .with_monthly_expenses({
            'mortgage': 2200,
            'food': 800,
            'childcare': 1200,
            'transportation': 500,
            'utilities': 200,
            'insurance': 400,
            'other': 400
        })
        .with_risk_tolerance(RiskTolerance.MODERATE)
        .build())
    
    print(f"Created custom profile: {custom_profile.name}")
    print(f"Net Worth: ${custom_profile.calculate_net_worth():,.0f}")
    print(f"Monthly Cash Flow: ${custom_profile.calculate_monthly_cash_flow():,.0f}")
    
    validation = ProfileValidator.validate_profile(custom_profile)
    print(f"Validation: {validation['completeness_score']:.1%} complete")
    
    if validation['recommendations']:
        print("Recommendations:")
        for rec in validation['recommendations'][:3]:
            print(f"  • {rec}")
    
    print("\n✅ Custom Profiles Demo Complete!")
