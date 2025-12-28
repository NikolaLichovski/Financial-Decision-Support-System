from typing import Dict


class PreferenceEngine:
    """
    Translates user preferences into interpretive context that shapes LLM reasoning.
    This is the core of preference-driven DSS - preferences actively modulate
    how information is framed, not just reported.
    """
    
    def __init__(self, risk_tolerance: str, time_horizon: str, risk_behavior: str):
        """
        Args:
            risk_tolerance: "Low", "Medium", or "High"
            time_horizon: "Short-term (<1yr)" or "Long-term (>1yr)"
            risk_behavior: "Risk-averse" or "Risk-seeking"
        """
        self.risk_tolerance = risk_tolerance
        self.time_horizon = time_horizon
        self.risk_behavior = risk_behavior
        
        # Preference matrices for interpretation
        self._init_interpretation_contexts()
    
    def _init_interpretation_contexts(self):
        """Initialize interpretation frameworks based on DSS principles"""
        
        # Volatility interpretation varies by risk profile
        self.volatility_frames = {
            "Low": {
                "emphasis": "downside protection and capital preservation",
                "concern_language": "risk exposure, potential losses, drawdown severity",
                "positive_language": "stability, predictability, preservation",
                "threshold_view": "conservative risk bounds"
            },
            "Medium": {
                "emphasis": "balanced growth with managed volatility",
                "concern_language": "portfolio fluctuations, risk-adjusted returns",
                "positive_language": "growth opportunities, reasonable stability",
                "threshold_view": "moderate risk tolerance"
            },
            "High": {
                "emphasis": "return potential and growth opportunities",
                "concern_language": "opportunity cost, market dynamics",
                "positive_language": "upside capture, aggressive growth, market opportunities",
                "threshold_view": "elevated risk acceptance"
            }
        }
        
        # Time horizon affects data focus
        self.time_frames = {
            "Short-term (<1yr)": {
                "data_focus": "recent 3-6 month trends and near-term momentum",
                "volatility_impact": "near-term price fluctuations and liquidity",
                "recovery_concern": "short recovery windows are critical",
                "relevant_metrics": "recent performance, current trends"
            },
            "Long-term (>1yr)": {
                "data_focus": "multi-year patterns and fundamental stability",
                "volatility_impact": "long-term trajectory smooths short-term noise",
                "recovery_concern": "extended recovery periods are acceptable",
                "relevant_metrics": "sustained trends, sector fundamentals"
            }
        }
        
        # Risk behavior modulates interpretation tone
        self.behavior_frames = {
            "Risk-averse": {
                "perspective": "conservative with emphasis on protection",
                "decision_frame": "what could go wrong and how to avoid losses",
                "uncertainty_view": "threats to be mitigated",
                "trade_off_priority": "safety over growth"
            },
            "Risk-seeking": {
                "perspective": "opportunistic with emphasis on potential",
                "decision_frame": "what upside exists and how to capture gains",
                "uncertainty_view": "opportunities to be seized",
                "trade_off_priority": "growth over safety"
            }
        }
    
    def get_interpretive_context(self) -> Dict[str, str]:
        """
        Generate interpretive context that will shape LLM reasoning.
        Returns structured guidance for how to frame analysis.
        """
        vol_frame = self.volatility_frames[self.risk_tolerance]
        time_frame = self.time_frames[self.time_horizon]
        behavior_frame = self.behavior_frames[self.risk_behavior]
        
        return {
            "volatility_emphasis": vol_frame["emphasis"],
            "concern_language": vol_frame["concern_language"],
            "positive_language": vol_frame["positive_language"],
            "data_focus": time_frame["data_focus"],
            "volatility_interpretation": time_frame["volatility_impact"],
            "recovery_perspective": time_frame["recovery_concern"],
            "analysis_perspective": behavior_frame["perspective"],
            "decision_framing": behavior_frame["decision_frame"],
            "trade_off_priority": behavior_frame["trade_off_priority"]
        }
    
    def get_prompt_guidance(self) -> str:
        """
        Generate explicit prompt guidance for the LLM that operationalizes preferences.
        This is injected into the system prompt to actively shape reasoning.
        """
        context = self.get_interpretive_context()
        
        guidance = f"""
PREFERENCE-DRIVEN ANALYSIS GUIDANCE:

Risk Profile Context:
- The user has {self.risk_tolerance.lower()} risk tolerance with {self.risk_behavior.lower()} behavior
- Frame volatility and uncertainty in terms of: {context['volatility_emphasis']}
- When discussing risks, emphasize: {context['concern_language']}
- When discussing opportunities, emphasize: {context['positive_language']}
- Trade-off priority: {context['trade_off_priority']}

Time Horizon Context:
- Investment horizon: {self.time_horizon.lower()}
- Focus analysis on: {context['data_focus']}
- Interpret volatility as: {context['volatility_interpretation']}
- Recovery time perspective: {context['recovery_perspective']}

Analysis Perspective:
- Adopt a {context['analysis_perspective']} viewpoint
- Frame decision considerations around: {context['decision_framing']}

CRITICAL: These preferences should shape HOW you interpret and present data,
not just be restated. For example, the same 25% volatility should be framed
as "significant downside risk" for risk-averse users but "opportunity for
outsized returns" for risk-seeking users. The data is the same; the
interpretation changes based on user context.
"""
        return guidance
    
    def interpret_risk_metric(self, metric_name: str, value: float) -> str:
        """
        Provide preference-specific interpretation of a risk metric
        
        Args:
            metric_name: e.g., "volatility", "drawdown", "beta"
            value: numeric value of the metric
            
        Returns:
            Interpreted description based on preferences
        """
        if metric_name == "volatility":
            return self._interpret_volatility(value)
        elif metric_name == "drawdown":
            return self._interpret_drawdown(value)
        elif metric_name == "beta":
            return self._interpret_beta(value)
        else:
            return f"{metric_name}: {value}"
    
    def _interpret_volatility(self, vol: float) -> str:
        """Interpret volatility based on risk preferences"""
        vol_frame = self.volatility_frames[self.risk_tolerance]
        
        if self.risk_tolerance == "Low":
            if vol < 15:
                return f"{vol}% volatility indicates stable, predictable behavior aligned with conservative objectives"
            elif vol < 25:
                return f"{vol}% volatility suggests price swings that may exceed comfort thresholds for capital preservation"
            else:
                return f"{vol}% volatility represents substantial fluctuation risk unsuitable for conservative portfolios"
        
        elif self.risk_tolerance == "High":
            if vol < 15:
                return f"{vol}% volatility suggests limited price movement, constraining potential for aggressive returns"
            elif vol < 25:
                return f"{vol}% volatility provides meaningful opportunity for returns while remaining investable"
            else:
                return f"{vol}% volatility creates significant return potential during favorable market phases"
        
        else:  # Medium
            if vol < 15:
                return f"{vol}% volatility indicates low-risk behavior suitable for core holdings"
            elif vol < 25:
                return f"{vol}% volatility is typical for balanced growth strategies"
            else:
                return f"{vol}% volatility exceeds typical balanced portfolio guidelines"
    
    def _interpret_drawdown(self, dd: float) -> str:
        """Interpret maximum drawdown"""
        dd_abs = abs(dd)
        
        if self.risk_behavior == "Risk-averse":
            if dd_abs < 10:
                return f"{dd:.1f}% maximum drawdown indicates limited downside exposure"
            elif dd_abs < 20:
                return f"{dd:.1f}% maximum drawdown represents notable capital risk requiring consideration"
            else:
                return f"{dd:.1f}% maximum drawdown signals severe downside exposure posing preservation challenges"
        else:  # Risk-seeking
            if dd_abs < 10:
                return f"{dd:.1f}% maximum drawdown suggests constrained volatility limiting return potential"
            elif dd_abs < 20:
                return f"{dd:.1f}% maximum drawdown is typical for growth-oriented investments"
            else:
                return f"{dd:.1f}% maximum drawdown indicates high volatility characteristic of aggressive positions"
    
    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta (market sensitivity)"""
        if beta < 0.8:
            sensitivity = "below-market volatility"
        elif beta < 1.2:
            sensitivity = "market-like volatility"
        else:
            sensitivity = "above-market volatility"
        
        if self.risk_behavior == "Risk-averse":
            if beta < 0.8:
                return f"Beta of {beta:.2f} indicates {sensitivity}, providing defensive characteristics"
            elif beta < 1.2:
                return f"Beta of {beta:.2f} indicates {sensitivity}, tracking market movements closely"
            else:
                return f"Beta of {beta:.2f} indicates {sensitivity}, amplifying market downturns"
        else:
            if beta < 0.8:
                return f"Beta of {beta:.2f} indicates {sensitivity}, limiting upside capture potential"
            elif beta < 1.2:
                return f"Beta of {beta:.2f} indicates {sensitivity}, participating in market gains proportionally"
            else:
                return f"Beta of {beta:.2f} indicates {sensitivity}, amplifying market upside"
    
    def get_preference_summary(self) -> str:
        """Return human-readable summary of preferences"""
        return f"""
User Preference Profile:
- Risk Tolerance: {self.risk_tolerance}
- Time Horizon: {self.time_horizon}
- Risk Behavior: {self.risk_behavior}
"""