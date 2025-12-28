import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


class FinancialDataProvider:
    """Fetches and summarizes stock data for DSS analysis"""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_summary(self, ticker: str, period: str = "1y") -> Optional[Dict]:
        """
        Fetch and compute comprehensive stock summary
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y)
            
        Returns:
            Dictionary with structured summary or None if fetch fails
        """
        try:
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            info = stock.info
            
            if hist.empty:
                return None
            
            # Compute metrics
            risk_metrics = self._compute_risk_metrics(hist)
            performance_metrics = self._compute_performance(hist)
            trend_analysis = self._analyze_trends(hist)
            
            # Build summary structure
            summary = {
                "ticker": ticker.upper(),
                "period": period,
                "fetch_date": datetime.now().strftime("%Y-%m-%d"),
                "basic_info": {
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "current_price": round(hist['Close'].iloc[-1], 2),
                    "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
                    "market_cap": info.get("marketCap", "N/A")
                },
                "risk_metrics": risk_metrics,
                "performance": performance_metrics,
                "trends": trend_analysis
            }
            
            return summary
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _compute_risk_metrics(self, hist: pd.DataFrame) -> Dict:
        """Compute volatility, drawdown, and risk indicators"""
        returns = hist['Close'].pct_change().dropna()
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Beta (approximate using variance if SPY data available)
        try:
            spy = yf.Ticker("SPY").history(period="1y")['Close']
            market_returns = spy.pct_change().dropna()
            
            # Align dates
            aligned_returns = returns.reindex(market_returns.index).dropna()
            aligned_market = market_returns.reindex(aligned_returns.index).dropna()
            
            covariance = np.cov(aligned_returns, aligned_market)[0][1]
            market_variance = np.var(aligned_market)
            beta = covariance / market_variance if market_variance > 0 else 1.0
        except:
            beta = None
        
        # Drawdown recovery time
        recovery_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for i, val in enumerate(drawdown):
            if val < -0.05 and not in_drawdown:  # Entered 5%+ drawdown
                in_drawdown = True
                drawdown_start = i
            elif val >= -0.01 and in_drawdown:  # Recovered to within 1%
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_periods.append(i - drawdown_start)
        
        avg_recovery_days = int(np.mean(recovery_periods)) if recovery_periods else None
        
        # Risk classification
        if volatility < 15:
            risk_class = "Low"
        elif volatility < 25:
            risk_class = "Moderate"
        else:
            risk_class = "High"
        
        return {
            "volatility_annual": round(volatility, 1),
            "max_drawdown": round(max_drawdown, 1),
            "beta": round(beta, 2) if beta else None,
            "avg_recovery_days": avg_recovery_days,
            "risk_classification": risk_class,
            "sharp_moves_count": len([r for r in returns if abs(r) > 0.05])  # Days with 5%+ moves
        }
    
    def _compute_performance(self, hist: pd.DataFrame) -> Dict:
        """Compute return metrics"""
        prices = hist['Close']
        
        # Various time periods
        periods = {
            "1_month": 21,
            "3_months": 63,
            "6_months": 126,
            "1_year": 252
        }
        
        returns = {}
        for name, days in periods.items():
            if len(prices) >= days:
                period_return = ((prices.iloc[-1] / prices.iloc[-days]) - 1) * 100
                returns[name] = round(period_return, 2)
            else:
                returns[name] = None
        
        # Compare to SPY (market benchmark)
        try:
            spy = yf.Ticker("SPY").history(period="1y")['Close']
            if len(spy) >= 252 and len(prices) >= 252:
                stock_annual = ((prices.iloc[-1] / prices.iloc[-252]) - 1) * 100
                spy_annual = ((spy.iloc[-1] / spy.iloc[-252]) - 1) * 100
                vs_market = round(stock_annual - spy_annual, 2)
            else:
                vs_market = None
        except:
            vs_market = None
        
        return {
            "return_1m": returns["1_month"],
            "return_3m": returns["3_months"],
            "return_6m": returns["6_months"],
            "return_1y": returns["1_year"],
            "vs_sp500_1y": vs_market
        }
    
    def _analyze_trends(self, hist: pd.DataFrame) -> Dict:
        """Analyze price and volume trends"""
        prices = hist['Close']
        volume = hist['Volume']
        
        # Recent trend (last 3 months vs prior 3 months)
        if len(prices) >= 126:
            recent_avg = prices.iloc[-63:].mean()
            prior_avg = prices.iloc[-126:-63].mean()
            
            if recent_avg > prior_avg * 1.05:
                price_trend = "rising"
            elif recent_avg < prior_avg * 0.95:
                price_trend = "declining"
            else:
                price_trend = "stable"
        else:
            price_trend = "insufficient_data"
        
        # Volume trend
        if len(volume) >= 63:
            recent_vol = volume.iloc[-21:].mean()
            prior_vol = volume.iloc[-63:-21].mean()
            
            if recent_vol > prior_vol * 1.2:
                volume_trend = "increasing"
            elif recent_vol < prior_vol * 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_trend = "insufficient_data"
        
        # Simple moving average position
        if len(prices) >= 50:
            sma_50 = prices.iloc[-50:].mean()
            current = prices.iloc[-1]
            
            if current > sma_50 * 1.02:
                sma_position = "above"
            elif current < sma_50 * 0.98:
                sma_position = "below"
            else:
                sma_position = "near"
        else:
            sma_position = "insufficient_data"
        
        return {
            "price_trend_3m": price_trend,
            "volume_trend": volume_trend,
            "position_vs_50day_avg": sma_position
        }
    
    def format_for_llm(self, summary: Dict, preferences: Dict) -> str:
        """
        Convert structured summary into narrative text optimized for LLM context
        
        Args:
            summary: Stock summary dictionary
            preferences: User preference dictionary for emphasis
            
        Returns:
            Formatted narrative string
        """
        if not summary:
            return "Unable to fetch stock data."
        
        # Build narrative sections
        sections = []
        
        # Header
        sections.append(f"=== STOCK PROFILE: {summary['ticker']} ===")
        sections.append(f"Analysis Period: {summary['period']} ending {summary['fetch_date']}\n")
        
        # Basic Information
        basic = summary['basic_info']
        sections.append("BASIC INFORMATION:")
        sections.append(f"  Sector: {basic['sector']}")
        sections.append(f"  Industry: {basic['industry']}")
        sections.append(f"  Current Price: ${basic['current_price']}")
        if basic['dividend_yield'] > 0:
            sections.append(f"  Dividend Yield: {basic['dividend_yield']}%")
        sections.append("")
        
        # Risk Metrics (emphasis based on preferences)
        risk = summary['risk_metrics']
        sections.append("RISK CHARACTERISTICS:")
        sections.append(f"  Annualized Volatility: {risk['volatility_annual']}% ({risk['risk_classification']} risk)")
        
        # Frame volatility based on risk tolerance
        vol_context = self._get_volatility_context(
            risk['volatility_annual'], 
            preferences.get('risk_tolerance', 'Medium')
        )
        sections.append(f"  Context: {vol_context}")
        
        sections.append(f"  Maximum Drawdown (period): {risk['max_drawdown']}%")
        
        if risk['beta']:
            sections.append(f"  Beta (market sensitivity): {risk['beta']}")
        
        if risk['avg_recovery_days']:
            sections.append(f"  Average Recovery Time: {risk['avg_recovery_days']} days")
        
        sections.append(f"  Sharp Moves (>5%): {risk['sharp_moves_count']} days")
        sections.append("")
        
        # Performance
        perf = summary['performance']
        sections.append("HISTORICAL PERFORMANCE:")
        
        if perf['return_1m'] is not None:
            sections.append(f"  1-Month Return: {perf['return_1m']:+.2f}%")
        if perf['return_3m'] is not None:
            sections.append(f"  3-Month Return: {perf['return_3m']:+.2f}%")
        if perf['return_1y'] is not None:
            sections.append(f"  1-Year Return: {perf['return_1y']:+.2f}%")
        
        if perf['vs_sp500_1y'] is not None:
            benchmark_text = "outperformed" if perf['vs_sp500_1y'] > 0 else "underperformed"
            sections.append(f"  vs. S&P 500: {benchmark_text} by {abs(perf['vs_sp500_1y']):.2f}%")
        sections.append("")
        
        # Trends
        trends = summary['trends']
        sections.append("RECENT TRENDS:")
        sections.append(f"  3-Month Price Trend: {trends['price_trend_3m']}")
        sections.append(f"  Volume Trend: {trends['volume_trend']}")
        sections.append(f"  Position vs 50-Day Average: {trends['position_vs_50day_avg']}")
        
        return "\n".join(sections)
    
    def _get_volatility_context(self, volatility: float, risk_tolerance: str) -> str:
        """Provide context for volatility based on risk tolerance"""
        if risk_tolerance == "Low":
            if volatility < 15:
                return "This low volatility suggests stable price behavior suitable for conservative portfolios"
            elif volatility < 25:
                return "This moderate volatility indicates notable price fluctuations that may exceed conservative risk thresholds"
            else:
                return "This high volatility represents substantial price swings and significant downside risk"
        
        elif risk_tolerance == "High":
            if volatility < 15:
                return "This low volatility limits potential for outsized returns but provides stability"
            elif volatility < 25:
                return "This moderate volatility offers balanced opportunity for returns with manageable swings"
            else:
                return "This high volatility creates opportunities for significant returns during favorable market conditions"
        
        else:  # Medium
            if volatility < 15:
                return "This low volatility provides predictable behavior with limited downside"
            elif volatility < 25:
                return "This moderate volatility is typical for diversified portfolios seeking balanced growth"
            else:
                return "This high volatility exceeds typical balanced portfolio thresholds"