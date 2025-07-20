"""
Volatility Surface Builder for Options Analysis
Builds and analyzes implied volatility surfaces for options trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy.interpolate import interp2d, griddata, RBFInterpolator
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VolatilitySurface:
    """Container for volatility surface data"""
    symbol: str
    spot_price: float
    surface_date: datetime
    strikes: np.ndarray
    expiries: np.ndarray
    implied_vols: np.ndarray  # 2D array of IVs
    interpolated_surface: Any  # Interpolation function
    surface_metrics: Dict[str, float]
    term_structure: pd.DataFrame
    skew_data: pd.DataFrame
    

@dataclass
class VolatilityMetrics:
    """Advanced volatility metrics"""
    atm_vol: float
    vol_of_vol: float
    skew_25d: float  # 25-delta skew
    skew_10d: float  # 10-delta skew
    term_structure_slope: float
    butterfly_25d: float
    risk_reversal_25d: float
    vol_smile_curvature: float
    vol_surface_smoothness: float


class VolatilitySurfaceBuilder:
    """
    Builds and analyzes implied volatility surfaces for options
    
    Features:
    - 3D volatility surface construction
    - Multiple interpolation methods
    - Volatility smile and skew analysis
    - Term structure analysis
    - Risk metrics calculation
    - Arbitrage detection
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.05,
                 interpolation_method: str = 'rbf',
                 min_moneyness: float = 0.7,
                 max_moneyness: float = 1.3):
        """Initialize volatility surface builder
        
        Args:
            risk_free_rate: Risk-free interest rate
            interpolation_method: Method for surface interpolation ('linear', 'cubic', 'rbf')
            min_moneyness: Minimum moneyness for surface
            max_moneyness: Maximum moneyness for surface
        """
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = risk_free_rate
        self.interpolation_method = interpolation_method
        self.min_moneyness = min_moneyness
        self.max_moneyness = max_moneyness
        
        # Greeks calculation parameters
        self.delta_shift = 0.01  # 1% shift for numerical derivatives
        
    def build_surface(self, 
                     options_data: pd.DataFrame,
                     spot_price: float,
                     surface_date: datetime) -> VolatilitySurface:
        """Build implied volatility surface from options data
        
        Args:
            options_data: DataFrame with columns: strike, expiry, option_type, iv, price
            spot_price: Current spot price
            surface_date: Date of the surface
            
        Returns:
            VolatilitySurface object
        """
        self.logger.info(f"Building volatility surface for spot={spot_price}")
        
        # Prepare data
        strikes, expiries, ivs = self._prepare_surface_data(options_data, spot_price)
        
        # Create interpolated surface
        interpolated_surface = self._create_interpolation(strikes, expiries, ivs)
        
        # Calculate surface metrics
        surface_metrics = self._calculate_surface_metrics(
            strikes, expiries, ivs, spot_price
        )
        
        # Extract term structure
        term_structure = self._extract_term_structure(
            options_data, spot_price
        )
        
        # Extract skew data
        skew_data = self._extract_skew_data(
            options_data, spot_price
        )
        
        # Get symbol from data
        symbol = options_data['symbol'].iloc[0] if 'symbol' in options_data.columns else 'UNKNOWN'
        
        return VolatilitySurface(
            symbol=symbol,
            spot_price=spot_price,
            surface_date=surface_date,
            strikes=strikes,
            expiries=expiries,
            implied_vols=ivs,
            interpolated_surface=interpolated_surface,
            surface_metrics=surface_metrics,
            term_structure=term_structure,
            skew_data=skew_data
        )
    
    def calculate_volatility_metrics(self, surface: VolatilitySurface) -> VolatilityMetrics:
        """Calculate advanced volatility metrics from surface
        
        Args:
            surface: VolatilitySurface object
            
        Returns:
            VolatilityMetrics object
        """
        # ATM volatility (shortest expiry)
        atm_vol = self._get_atm_vol(surface, expiry_idx=0)
        
        # Volatility of volatility
        vol_of_vol = self._calculate_vol_of_vol(surface)
        
        # Skew metrics
        skew_25d = self._calculate_delta_skew(surface, delta=0.25)
        skew_10d = self._calculate_delta_skew(surface, delta=0.10)
        
        # Term structure slope
        term_slope = self._calculate_term_structure_slope(surface)
        
        # Butterfly and risk reversal
        butterfly_25d = self._calculate_butterfly(surface, delta=0.25)
        rr_25d = self._calculate_risk_reversal(surface, delta=0.25)
        
        # Smile curvature
        smile_curvature = self._calculate_smile_curvature(surface)
        
        # Surface smoothness
        smoothness = self._calculate_surface_smoothness(surface)
        
        return VolatilityMetrics(
            atm_vol=atm_vol,
            vol_of_vol=vol_of_vol,
            skew_25d=skew_25d,
            skew_10d=skew_10d,
            term_structure_slope=term_slope,
            butterfly_25d=butterfly_25d,
            risk_reversal_25d=rr_25d,
            vol_smile_curvature=smile_curvature,
            vol_surface_smoothness=smoothness
        )
    
    def _prepare_surface_data(self, 
                            options_data: pd.DataFrame,
                            spot_price: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for surface construction"""
        # Filter data
        filtered_data = options_data[
            (options_data['iv'] > 0) & 
            (options_data['iv'] < 2.0)  # Remove unrealistic IVs
        ].copy()
        
        # Calculate moneyness
        filtered_data['moneyness'] = filtered_data['strike'] / spot_price
        
        # Filter by moneyness range
        filtered_data = filtered_data[
            (filtered_data['moneyness'] >= self.min_moneyness) &
            (filtered_data['moneyness'] <= self.max_moneyness)
        ]
        
        # Calculate time to expiry in years
        if 'expiry' in filtered_data.columns:
            filtered_data['time_to_expiry'] = pd.to_datetime(filtered_data['expiry']).apply(
                lambda x: (x - datetime.now()).days / 365.0
            )
        else:
            # Assume days_to_expiry column exists
            filtered_data['time_to_expiry'] = filtered_data['days_to_expiry'] / 365.0
        
        # Create unique strikes and expiries
        unique_strikes = np.sort(filtered_data['strike'].unique())
        unique_expiries = np.sort(filtered_data['time_to_expiry'].unique())
        
        # Create 2D grid of implied volatilities
        iv_grid = np.zeros((len(unique_strikes), len(unique_expiries)))
        
        for i, strike in enumerate(unique_strikes):
            for j, expiry in enumerate(unique_expiries):
                mask = (filtered_data['strike'] == strike) & \
                       (filtered_data['time_to_expiry'] == expiry)
                
                if mask.any():
                    # Average IV for calls and puts at same strike/expiry
                    iv_grid[i, j] = filtered_data[mask]['iv'].mean()
                else:
                    iv_grid[i, j] = np.nan
        
        # Fill NaN values using interpolation
        iv_grid = self._fill_missing_ivs(iv_grid)
        
        return unique_strikes, unique_expiries, iv_grid
    
    def _create_interpolation(self, 
                            strikes: np.ndarray,
                            expiries: np.ndarray,
                            ivs: np.ndarray) -> Any:
        """Create interpolation function for the surface"""
        if self.interpolation_method == 'linear':
            return interp2d(strikes, expiries, ivs.T, kind='linear')
        elif self.interpolation_method == 'cubic':
            return interp2d(strikes, expiries, ivs.T, kind='cubic')
        elif self.interpolation_method == 'rbf':
            # Prepare data for RBF interpolation
            points = []
            values = []
            
            for i, strike in enumerate(strikes):
                for j, expiry in enumerate(expiries):
                    if not np.isnan(ivs[i, j]):
                        points.append([strike, expiry])
                        values.append(ivs[i, j])
            
            points = np.array(points)
            values = np.array(values)
            
            # Create RBF interpolator
            return RBFInterpolator(points, values, smoothing=0.1)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
    
    def _fill_missing_ivs(self, iv_grid: np.ndarray) -> np.ndarray:
        """Fill missing IV values using interpolation"""
        # First, forward fill along strikes
        for j in range(iv_grid.shape[1]):
            col = iv_grid[:, j]
            mask = ~np.isnan(col)
            if mask.any():
                col[np.isnan(col)] = np.interp(
                    np.where(np.isnan(col))[0],
                    np.where(mask)[0],
                    col[mask]
                )
                iv_grid[:, j] = col
        
        # Then, forward fill along expiries
        for i in range(iv_grid.shape[0]):
            row = iv_grid[i, :]
            mask = ~np.isnan(row)
            if mask.any():
                row[np.isnan(row)] = np.interp(
                    np.where(np.isnan(row))[0],
                    np.where(mask)[0],
                    row[mask]
                )
                iv_grid[i, :] = row
        
        # Fill any remaining NaNs with average
        iv_grid[np.isnan(iv_grid)] = np.nanmean(iv_grid)
        
        return iv_grid
    
    def _calculate_surface_metrics(self,
                                 strikes: np.ndarray,
                                 expiries: np.ndarray,
                                 ivs: np.ndarray,
                                 spot_price: float) -> Dict[str, float]:
        """Calculate metrics describing the surface characteristics"""
        metrics = {}
        
        # Average IV across surface
        metrics['avg_iv'] = np.mean(ivs)
        metrics['iv_std'] = np.std(ivs)
        
        # Find ATM index
        atm_idx = np.argmin(np.abs(strikes - spot_price))
        
        # ATM term structure slope
        if len(expiries) > 1:
            atm_ivs = ivs[atm_idx, :]
            metrics['atm_term_slope'] = (atm_ivs[-1] - atm_ivs[0]) / (expiries[-1] - expiries[0])
        
        # Average skew (difference between OTM puts and calls)
        if len(strikes) > 2:
            put_wing_idx = max(0, atm_idx - len(strikes) // 4)
            call_wing_idx = min(len(strikes) - 1, atm_idx + len(strikes) // 4)
            
            avg_put_iv = np.mean(ivs[put_wing_idx, :])
            avg_call_iv = np.mean(ivs[call_wing_idx, :])
            metrics['avg_skew'] = avg_put_iv - avg_call_iv
        
        # Surface roughness (variation in IV)
        if ivs.shape[0] > 1 and ivs.shape[1] > 1:
            # Calculate gradient magnitude
            dy, dx = np.gradient(ivs)
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            metrics['surface_roughness'] = np.mean(gradient_magnitude)
        
        return metrics
    
    def _extract_term_structure(self,
                               options_data: pd.DataFrame,
                               spot_price: float) -> pd.DataFrame:
        """Extract ATM term structure from options data"""
        # Filter ATM options (within 2% of spot)
        atm_data = options_data[
            (options_data['strike'] >= spot_price * 0.98) &
            (options_data['strike'] <= spot_price * 1.02)
        ].copy()
        
        if atm_data.empty:
            return pd.DataFrame()
        
        # Calculate time to expiry
        if 'expiry' in atm_data.columns:
            atm_data['time_to_expiry'] = pd.to_datetime(atm_data['expiry']).apply(
                lambda x: (x - datetime.now()).days / 365.0
            )
        else:
            atm_data['time_to_expiry'] = atm_data['days_to_expiry'] / 365.0
        
        # Group by expiry and average IV
        term_structure = atm_data.groupby('time_to_expiry').agg({
            'iv': 'mean',
            'strike': 'mean'
        }).reset_index()
        
        term_structure = term_structure.sort_values('time_to_expiry')
        
        return term_structure
    
    def _extract_skew_data(self,
                          options_data: pd.DataFrame,
                          spot_price: float) -> pd.DataFrame:
        """Extract volatility skew for each expiry"""
        skew_data = []
        
        # Get unique expiries
        if 'time_to_expiry' not in options_data.columns:
            if 'expiry' in options_data.columns:
                options_data['time_to_expiry'] = pd.to_datetime(options_data['expiry']).apply(
                    lambda x: (x - datetime.now()).days / 365.0
                )
            else:
                options_data['time_to_expiry'] = options_data['days_to_expiry'] / 365.0
        
        unique_expiries = options_data['time_to_expiry'].unique()
        
        for expiry in unique_expiries:
            expiry_data = options_data[options_data['time_to_expiry'] == expiry]
            
            if len(expiry_data) < 3:
                continue
            
            # Calculate moneyness
            expiry_data['moneyness'] = expiry_data['strike'] / spot_price
            
            # Sort by strike
            expiry_data = expiry_data.sort_values('strike')
            
            # Calculate skew metrics
            atm_iv = expiry_data[
                expiry_data['moneyness'].between(0.98, 1.02)
            ]['iv'].mean()
            
            otm_put_iv = expiry_data[
                expiry_data['moneyness'] < 0.95
            ]['iv'].mean()
            
            otm_call_iv = expiry_data[
                expiry_data['moneyness'] > 1.05
            ]['iv'].mean()
            
            skew_data.append({
                'expiry': expiry,
                'atm_iv': atm_iv,
                'put_skew': otm_put_iv - atm_iv if not np.isnan(otm_put_iv) else 0,
                'call_skew': otm_call_iv - atm_iv if not np.isnan(otm_call_iv) else 0,
                'total_skew': (otm_put_iv - otm_call_iv) if not (np.isnan(otm_put_iv) or np.isnan(otm_call_iv)) else 0
            })
        
        return pd.DataFrame(skew_data)
    
    def _get_atm_vol(self, surface: VolatilitySurface, expiry_idx: int = 0) -> float:
        """Get ATM implied volatility"""
        atm_idx = np.argmin(np.abs(surface.strikes - surface.spot_price))
        return surface.implied_vols[atm_idx, expiry_idx]
    
    def _calculate_vol_of_vol(self, surface: VolatilitySurface) -> float:
        """Calculate volatility of implied volatility"""
        # Calculate changes in ATM vol across time
        atm_idx = np.argmin(np.abs(surface.strikes - surface.spot_price))
        atm_vols = surface.implied_vols[atm_idx, :]
        
        if len(atm_vols) > 1:
            vol_changes = np.diff(atm_vols)
            return np.std(vol_changes)
        
        return 0.0
    
    def _calculate_delta_skew(self, surface: VolatilitySurface, delta: float) -> float:
        """Calculate skew at specific delta"""
        # Find strikes corresponding to delta
        # This is simplified - in practice would use proper option pricing model
        
        # Approximate strike for delta using Black-Scholes approximation
        # For 25-delta put: K ≈ S * exp(-0.675 * σ * √T)
        # For 25-delta call: K ≈ S * exp(0.675 * σ * √T)
        
        shortest_expiry = surface.expiries[0]
        atm_vol = self._get_atm_vol(surface)
        
        put_strike = surface.spot_price * np.exp(-norm.ppf(1-delta) * atm_vol * np.sqrt(shortest_expiry))
        call_strike = surface.spot_price * np.exp(norm.ppf(1-delta) * atm_vol * np.sqrt(shortest_expiry))
        
        # Find closest strikes in surface
        put_idx = np.argmin(np.abs(surface.strikes - put_strike))
        call_idx = np.argmin(np.abs(surface.strikes - call_strike))
        
        put_vol = surface.implied_vols[put_idx, 0]
        call_vol = surface.implied_vols[call_idx, 0]
        
        return put_vol - call_vol
    
    def _calculate_term_structure_slope(self, surface: VolatilitySurface) -> float:
        """Calculate slope of ATM term structure"""
        if len(surface.term_structure) < 2:
            return 0.0
        
        # Linear regression on ATM vols vs time
        x = surface.term_structure['time_to_expiry'].values
        y = surface.term_structure['iv'].values
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        return 0.0
    
    def _calculate_butterfly(self, surface: VolatilitySurface, delta: float) -> float:
        """Calculate butterfly spread (convexity measure)"""
        # Butterfly = (25d Put IV + 25d Call IV) / 2 - ATM IV
        
        skew = self._calculate_delta_skew(surface, delta)
        atm_vol = self._get_atm_vol(surface)
        
        # Get individual wing vols
        shortest_expiry = surface.expiries[0]
        put_strike = surface.spot_price * np.exp(-norm.ppf(1-delta) * atm_vol * np.sqrt(shortest_expiry))
        call_strike = surface.spot_price * np.exp(norm.ppf(1-delta) * atm_vol * np.sqrt(shortest_expiry))
        
        put_idx = np.argmin(np.abs(surface.strikes - put_strike))
        call_idx = np.argmin(np.abs(surface.strikes - call_strike))
        
        put_vol = surface.implied_vols[put_idx, 0]
        call_vol = surface.implied_vols[call_idx, 0]
        
        return (put_vol + call_vol) / 2 - atm_vol
    
    def _calculate_risk_reversal(self, surface: VolatilitySurface, delta: float) -> float:
        """Calculate risk reversal (directional bias)"""
        # Risk Reversal = 25d Call IV - 25d Put IV
        return -self._calculate_delta_skew(surface, delta)
    
    def _calculate_smile_curvature(self, surface: VolatilitySurface) -> float:
        """Calculate curvature of volatility smile"""
        # Use shortest expiry
        smile = surface.implied_vols[:, 0]
        strikes = surface.strikes
        
        # Fit quadratic to smile
        if len(strikes) > 2:
            # Normalize strikes by spot
            normalized_strikes = strikes / surface.spot_price
            coeffs = np.polyfit(normalized_strikes, smile, 2)
            
            # Return quadratic coefficient (curvature)
            return coeffs[0]
        
        return 0.0
    
    def _calculate_surface_smoothness(self, surface: VolatilitySurface) -> float:
        """Calculate smoothness of the volatility surface"""
        # Use second derivatives as measure of smoothness
        ivs = surface.implied_vols
        
        if ivs.shape[0] > 2 and ivs.shape[1] > 2:
            # Calculate second derivatives
            d2_strike = np.diff(ivs, n=2, axis=0)
            d2_expiry = np.diff(ivs, n=2, axis=1)
            
            # RMS of second derivatives (lower = smoother)
            smoothness = 1.0 / (1.0 + np.sqrt(np.mean(d2_strike**2) + np.mean(d2_expiry**2)))
            return smoothness
        
        return 0.5
    
    def detect_arbitrage(self, surface: VolatilitySurface) -> List[Dict[str, Any]]:
        """Detect potential arbitrage opportunities in the surface"""
        arbitrage_opportunities = []
        
        # Check for calendar spread arbitrage
        calendar_arb = self._check_calendar_arbitrage(surface)
        if calendar_arb:
            arbitrage_opportunities.extend(calendar_arb)
        
        # Check for butterfly arbitrage
        butterfly_arb = self._check_butterfly_arbitrage(surface)
        if butterfly_arb:
            arbitrage_opportunities.extend(butterfly_arb)
        
        # Check for vertical spread arbitrage
        vertical_arb = self._check_vertical_arbitrage(surface)
        if vertical_arb:
            arbitrage_opportunities.extend(vertical_arb)
        
        return arbitrage_opportunities
    
    def _check_calendar_arbitrage(self, surface: VolatilitySurface) -> List[Dict[str, Any]]:
        """Check for calendar spread arbitrage (IV should increase with time)"""
        opportunities = []
        
        # Check each strike
        for i, strike in enumerate(surface.strikes):
            ivs_by_expiry = surface.implied_vols[i, :]
            
            # Check if IV decreases with time
            for j in range(len(ivs_by_expiry) - 1):
                if ivs_by_expiry[j] > ivs_by_expiry[j + 1]:
                    opportunities.append({
                        'type': 'calendar_arbitrage',
                        'strike': strike,
                        'near_expiry': surface.expiries[j],
                        'far_expiry': surface.expiries[j + 1],
                        'near_iv': ivs_by_expiry[j],
                        'far_iv': ivs_by_expiry[j + 1],
                        'iv_difference': ivs_by_expiry[j] - ivs_by_expiry[j + 1]
                    })
        
        return opportunities
    
    def _check_butterfly_arbitrage(self, surface: VolatilitySurface) -> List[Dict[str, Any]]:
        """Check for butterfly arbitrage (convexity violations)"""
        opportunities = []
        
        # Check each expiry
        for j, expiry in enumerate(surface.expiries):
            strikes = surface.strikes
            ivs = surface.implied_vols[:, j]
            
            # Check butterfly condition: IV(K2) <= 0.5 * (IV(K1) + IV(K3))
            for i in range(1, len(strikes) - 1):
                k1, k2, k3 = strikes[i-1], strikes[i], strikes[i+1]
                iv1, iv2, iv3 = ivs[i-1], ivs[i], ivs[i+1]
                
                max_iv2 = 0.5 * (iv1 + iv3)
                
                if iv2 > max_iv2 * 1.01:  # 1% tolerance
                    opportunities.append({
                        'type': 'butterfly_arbitrage',
                        'expiry': expiry,
                        'strikes': [k1, k2, k3],
                        'ivs': [iv1, iv2, iv3],
                        'violation': iv2 - max_iv2
                    })
        
        return opportunities
    
    def _check_vertical_arbitrage(self, surface: VolatilitySurface) -> List[Dict[str, Any]]:
        """Check for vertical spread arbitrage"""
        opportunities = []
        
        # For each expiry, check if option prices decrease with strike for calls
        # This is a simplified check - in practice would calculate actual option prices
        
        return opportunities
    
    def get_trading_signals(self, metrics: VolatilityMetrics) -> Dict[str, Any]:
        """Generate trading signals based on volatility metrics"""
        signals = {
            'vol_regime': self._classify_vol_regime(metrics),
            'skew_signal': self._get_skew_signal(metrics),
            'term_structure_signal': self._get_term_structure_signal(metrics),
            'surface_signal': self._get_surface_signal(metrics),
            'recommendations': []
        }
        
        # Generate specific recommendations
        if metrics.skew_25d > 0.05:  # Significant put skew
            signals['recommendations'].append({
                'strategy': 'put_spread',
                'reason': 'High put skew indicates downside protection demand',
                'confidence': min(metrics.skew_25d * 10, 1.0)
            })
        
        if metrics.term_structure_slope > 0.1:  # Steep term structure
            signals['recommendations'].append({
                'strategy': 'calendar_spread',
                'reason': 'Steep term structure favors selling near-term vol',
                'confidence': min(metrics.term_structure_slope * 5, 1.0)
            })
        
        if metrics.butterfly_25d > 0.02:  # High convexity
            signals['recommendations'].append({
                'strategy': 'butterfly',
                'reason': 'High smile convexity suggests range-bound market',
                'confidence': min(metrics.butterfly_25d * 20, 1.0)
            })
        
        return signals
    
    def _classify_vol_regime(self, metrics: VolatilityMetrics) -> str:
        """Classify current volatility regime"""
        if metrics.atm_vol < 0.15:
            return 'low_vol'
        elif metrics.atm_vol < 0.25:
            return 'normal_vol'
        elif metrics.atm_vol < 0.40:
            return 'high_vol'
        else:
            return 'extreme_vol'
    
    def _get_skew_signal(self, metrics: VolatilityMetrics) -> str:
        """Get signal from skew metrics"""
        if metrics.risk_reversal_25d < -0.03:
            return 'bearish_skew'
        elif metrics.risk_reversal_25d > 0.03:
            return 'bullish_skew'
        else:
            return 'neutral_skew'
    
    def _get_term_structure_signal(self, metrics: VolatilityMetrics) -> str:
        """Get signal from term structure"""
        if metrics.term_structure_slope > 0.1:
            return 'contango'  # Expecting vol to rise
        elif metrics.term_structure_slope < -0.05:
            return 'backwardation'  # Expecting vol to fall
        else:
            return 'flat'
    
    def _get_surface_signal(self, metrics: VolatilityMetrics) -> str:
        """Get overall surface signal"""
        if metrics.vol_surface_smoothness < 0.3:
            return 'disrupted'  # Rough surface, potential mispricings
        elif metrics.vol_smile_curvature > 0.5:
            return 'convex'  # High convexity, range-bound expectations
        else:
            return 'normal'
    
    def interpolate_iv(self, 
                      surface: VolatilitySurface,
                      strike: float,
                      time_to_expiry: float) -> float:
        """Interpolate IV for any strike/expiry combination
        
        Args:
            surface: VolatilitySurface object
            strike: Target strike price
            time_to_expiry: Target time to expiry (years)
            
        Returns:
            Interpolated implied volatility
        """
        if isinstance(surface.interpolated_surface, RBFInterpolator):
            # RBF interpolator expects 2D input
            point = np.array([[strike, time_to_expiry]])
            return float(surface.interpolated_surface(point)[0])
        else:
            # interp2d returns a callable
            return float(surface.interpolated_surface(strike, time_to_expiry)[0])
    
    def calculate_local_volatility(self, 
                                 surface: VolatilitySurface,
                                 strike: float,
                                 time: float) -> float:
        """Calculate local volatility using Dupire formula
        
        This is a simplified implementation - full implementation would
        require more sophisticated numerical methods
        """
        # Numerical derivatives
        dk = surface.spot_price * 0.01  # 1% shift
        dt = 1/365  # 1 day
        
        # Get IVs at different points
        iv_center = self.interpolate_iv(surface, strike, time)
        iv_up = self.interpolate_iv(surface, strike + dk, time)
        iv_down = self.interpolate_iv(surface, strike - dk, time)
        iv_later = self.interpolate_iv(surface, strike, time + dt)
        
        # Calculate derivatives
        div_dk = (iv_up - iv_down) / (2 * dk)
        d2iv_dk2 = (iv_up - 2 * iv_center + iv_down) / (dk ** 2)
        div_dt = (iv_later - iv_center) / dt
        
        # Simplified Dupire formula
        # σ_local = sqrt(2 * ∂σ/∂T / (1 + 2*d*K*∂σ/∂K + K²*∂²σ/∂K²))
        # where d = drift term (simplified to 0 here)
        
        numerator = 2 * div_dt
        denominator = 1 + strike ** 2 * d2iv_dk2
        
        if denominator > 0 and numerator > 0:
            local_vol = np.sqrt(numerator / denominator)
            return local_vol
        else:
            # Fallback to implied vol
            return iv_center