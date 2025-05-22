import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math
from typing import Tuple, List, Optional
import talib

# Configuration and constants
class Config:
    # Environment parameters
    TICK_SPACING = 60  # For 0.3% fee pools
    MAX_WIDTH = 20     # Maximum liquidity interval width
    GAS_FEE = 1.0     # Flat gas fee assumption
    
    # DRL parameters
    HIDDEN_UNITS = [64, 64]
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 256
    BUFFER_SIZE = int(1e6)
    GAMMA = 0.9
    TARGET_UPDATE_RATE = 0.01
    GRADIENT_CLIP_NORM = 0.7
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995

# class TechnicalIndicators:
#     """Calculate technical indicators for state features"""
    
#     @staticmethod
#     def calculate_features(df: pd.DataFrame) -> np.ndarray:
#         """Calculate all technical indicators from OHLCV data"""
#         features = []
        
#         # Basic price features
#         features.append(df['open'].values[-1])  # Current open price
#         features.append(df['high'].values[-1] / df['open'].values[-1])  # High/Open ratio
#         features.append(df['low'].values[-1] / df['open'].values[-1])   # Low/Open ratio
#         features.append(df['close'].values[-1] / df['open'].values[-1]) # Close/Open ratio
#         features.append(df['volume'].values[-1])  # Trading volume
        
#         # Technical indicators
#         close_prices = df['close'].values
#         high_prices = df['high'].values
#         low_prices = df['low'].values
#         volume = df['volume'].values
        
#         # DEMA - Double Exponential Moving Average
#         dema = talib.DEMA(close_prices, timeperiod=14)
#         features.append(dema[-1] / df['open'].values[-1] if not np.isnan(dema[-1]) else 1.0)
        
#         # SAR - Parabolic SAR
#         sar = talib.SAR(high_prices, low_prices)
#         features.append(sar[-1] / df['open'].values[-1] if not np.isnan(sar[-1]) else 1.0)
        
#         # ADX - Average Directional Movement Index
#         adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
#         features.append(adx[-1] if not np.isnan(adx[-1]) else 50.0)
        
#         # APO - Absolute Price Oscillator
#         apo = talib.APO(close_prices)
#         features.append(apo[-1] if not np.isnan(apo[-1]) else 0.0)
        
#         # AROON - Aroon Oscillator
#         aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=14)
#         aroon_osc = aroon_up[-1] - aroon_down[-1] if not (np.isnan(aroon_up[-1]) or np.isnan(aroon_down[-1])) else 0.0
#         features.append(aroon_osc)
        
#         # BOP - Balance of Power
#         bop = talib.BOP(df['open'].values, high_prices, low_prices, close_prices)
#         features.append(bop[-1] if not np.isnan(bop[-1]) else 0.0)
        
#         # CCI - Commodity Channel Index
#         cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
#         features.extend([cci[-1] if not np.isnan(cci[-1]) else 0.0, cci[-2] if len(cci) > 1 and not np.isnan(cci[-2]) else 0.0])
        
#         # CMO - Chande Momentum Oscillator
#         cmo = talib.CMO(close_prices, timeperiod=14)
#         features.append(cmo[-1] if not np.isnan(cmo[-1]) else 0.0)
        
#         # DX - Directional Movement Index
#         dx = talib.DX(high_prices, low_prices, close_prices, timeperiod=14)
#         features.append(dx[-1] if not np.isnan(dx[-1]) else 0.0)
        
#         # MINUS_DM - Minus Directional Movement
#         minus_dm = talib.MINUS_DM(high_prices, low_prices, timeperiod=14)
#         features.append(minus_dm[-1] if not np.isnan(minus_dm[-1]) else 0.0)
        
#         # MOM - Momentum
#         mom = talib.MOM(close_prices, timeperiod=10)
#         features.append(mom[-1] if not np.isnan(mom[-1]) else 0.0)
        
#         # PLUS_DM - Plus Directional Movement
#         plus_dm = talib.PLUS_DM(high_prices, low_prices, timeperiod=14)
#         features.append(plus_dm[-1] if not np.isnan(plus_dm[-1]) else 0.0)
        
#         # TRIX
#         trix = talib.TRIX(close_prices, timeperiod=30)
#         features.append(trix[-1] if not np.isnan(trix[-1]) else 0.0)
        
#         # ULTOSC - Ultimate Oscillator
#         ult_osc = talib.ULTOSC(high_prices, low_prices, close_prices)
#         features.append(ult_osc[-1] if not np.isnan(ult_osc[-1]) else 50.0)
        
#         # Stochastic indicators
#         slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
#         features.extend([
#             slowk[-1] if not np.isnan(slowk[-1]) else 50.0,
#             slowd[-1] if not np.isnan(slowd[-1]) else 50.0,
#             slowk[-1] - slowd[-1] if not (np.isnan(slowk[-1]) or np.isnan(slowd[-1])) else 0.0
#         ])
        
#         # Fast Stochastic
#         fastk, fastd = talib.STOCHF(high_prices, low_prices, close_prices)
#         features.extend([
#             fastk[-1] if not np.isnan(fastk[-1]) else 50.0,
#             fastd[-1] if not np.isnan(fastd[-1]) else 50.0,
#             fastk[-1] - fastd[-1] if not (np.isnan(fastk[-1]) or np.isnan(fastd[-1])) else 0.0
#         ])
        
#         # ATR - Average True Range
#         atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
#         natr = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
#         trange = talib.TRANGE(high_prices, low_prices, close_prices)
#         features.extend([
#             natr[-1] if not np.isnan(natr[-1]) else 1.0,
#             trange[-1] if not np.isnan(trange[-1]) else 0.0
#         ])
        
#         # Hilbert Transform indicators
#         ht_dcperiod = talib.HT_DCPERIOD(close_prices)
#         ht_dcphase = talib.HT_DCPHASE(close_prices)
#         features.extend([
#             ht_dcperiod[-1] if not np.isnan(ht_dcperiod[-1]) else 15.0,
#             ht_dcphase[-1] if not np.isnan(ht_dcphase[-1]) else 0.0
#         ])
        
#         return np.array(features, dtype=np.float32)



class UniswapV3Environment:
    """Uniswap V3 liquidity provision environment"""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 1000.0, fee_tier: float = 0.003):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.fee_tier = fee_tier
        self.tick_spacing = Config.TICK_SPACING
        
        # State variables
        self.current_step = 0
        self.cash = initial_capital
        self.liquidity_value = 0.0
        self.liquidity_center = 0  # Central tick of liquidity position
        self.liquidity_width = 0   # Width of liquidity interval
        self.liquidity_units = 0.0 # Amount of liquidity units L
        
        # Position tracking
        self.position_active = False
        
    def price_to_tick(self, price: float) -> int:
        """Convert price to tick"""
        return int(math.log(price, 1.0001))
    
    def tick_to_price(self, tick: int) -> float:
        """Convert tick to price"""
        return 1.0001 ** tick
    
    def calculate_reserves(self, price: float, tick_lower: int, tick_upper: int, liquidity: float) -> Tuple[float, float]:
        """Calculate token reserves for a liquidity position"""
        price_lower = self.tick_to_price(tick_lower)
        price_upper = self.tick_to_price(tick_upper)
        sqrt_price = math.sqrt(price)
        sqrt_price_lower = math.sqrt(price_lower)
        sqrt_price_upper = math.sqrt(price_upper)
        
        if price <= price_lower:
            # All token X
            x = liquidity * (1/sqrt_price_lower - 1/sqrt_price_upper)
            y = 0
        elif price >= price_upper:
            # All token Y
            x = 0
            y = liquidity * (sqrt_price_upper - sqrt_price_lower)
        else:
            # Mixed position
            x = liquidity * (1/sqrt_price - 1/sqrt_price_upper)
            y = liquidity * (sqrt_price - sqrt_price_lower)
            
        return x, y
    
    def calculate_position_value(self, price: float) -> float:
        """Calculate current value of liquidity position"""
        if not self.position_active or self.liquidity_units == 0:
            return 0.0
            
        tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
        tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
        
        x, y = self.calculate_reserves(price, tick_lower, tick_upper, self.liquidity_units)
        return price * x + y
    
    def calculate_trading_fee(self, price_from: float, price_to: float) -> float:
        """Calculate trading fees earned during price movement"""
        if not self.position_active or self.liquidity_units == 0:
            return 0.0
            
        tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
        tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
        price_lower = self.tick_to_price(tick_lower)
        price_upper = self.tick_to_price(tick_upper)
        
        # Check if price movement is within liquidity range
        if price_from <= price_lower and price_to <= price_lower:
            return 0.0
        if price_from >= price_upper and price_to >= price_upper:
            return 0.0
            
        # Calculate fee based on price movement within range
        sqrt_price_from = max(math.sqrt(price_lower), min(math.sqrt(price_upper), math.sqrt(price_from)))
        sqrt_price_to = max(math.sqrt(price_lower), min(math.sqrt(price_upper), math.sqrt(price_to)))
        
        fee = 0.0
        if price_to > price_from:  # Upward price movement
            fee = (self.fee_tier / (1 - self.fee_tier)) * self.liquidity_units * (sqrt_price_to - sqrt_price_from)
        else:  # Downward price movement
            fee = (self.fee_tier / (1 - self.fee_tier)) * self.liquidity_units * (1/sqrt_price_from - 1/sqrt_price_to) * price_from
            
        return max(0.0, fee)
    
    def calculate_lvr(self, price_from: float, price_to: float) -> float:
        """Calculate Loss-Versus-Rebalancing"""
        if not self.position_active or self.liquidity_units == 0:
            return 0.0
            
        tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
        tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
        price_lower = self.tick_to_price(tick_lower)
        price_upper = self.tick_to_price(tick_upper)
        
        # LVR calculation based on equation (5) from the paper
        value_from = self.calculate_position_value(price_from)
        value_to = self.calculate_position_value(price_to)
        
        x_from, _ = self.calculate_reserves(price_from, tick_lower, tick_upper, self.liquidity_units)
        
        # LVR = Change in position value - rebalancing portfolio return
        lvr = (value_to - value_from) - x_from * (price_to - price_from)
        
        return lvr
    
    def get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step < 30:  # Need enough history for technical indicators
            # Return zeros for insufficient history
            market_features = np.zeros(28, dtype=np.float32)
        else:
            # Calculate technical indicators
            hist_data = self.data.iloc[max(0, self.current_step-100):self.current_step+1]
            market_features = TechnicalIndicators.calculate_features(hist_data)
            
        # Position state
        cash_normalized = self.cash / self.initial_capital
        center_tick_normalized = self.liquidity_center / 10000.0  # Normalize tick
        width_normalized = self.liquidity_width / Config.MAX_WIDTH
        position_value_normalized = self.liquidity_value / self.initial_capital
        
        state = np.concatenate([
            market_features,
            [cash_normalized, center_tick_normalized, width_normalized, position_value_normalized]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self.get_state(), 0.0, True, {}
            
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Calculate rewards before taking action
        trading_fee = 0.0
        lvr = 0.0
        gas_fee = 0.0
        
        if self.position_active:
            trading_fee = self.calculate_trading_fee(current_price, next_price)
            lvr = self.calculate_lvr(current_price, next_price)
        
        # Take action
        if action == 0:
            # Hold current position
            pass
        else:
            # Reallocate liquidity
            gas_fee = Config.GAS_FEE
            
            # Close current position
            if self.position_active:
                self.cash += self.calculate_position_value(current_price)
                self.position_active = False
            
            # Open new position
            current_tick = self.price_to_tick(current_price)
            self.liquidity_center = self.tick_spacing * round(current_tick / self.tick_spacing)
            self.liquidity_width = action
            
            # Calculate liquidity units based on available capital
            total_capital = self.cash + self.liquidity_value
            
            tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
            tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
            
            # Simplified liquidity calculation
            price_lower = self.tick_to_price(tick_lower)
            price_upper = self.tick_to_price(tick_upper)
            sqrt_price = math.sqrt(current_price)
            sqrt_price_lower = math.sqrt(price_lower)
            sqrt_price_upper = math.sqrt(price_upper)
            
            if current_price <= price_lower:
                self.liquidity_units = total_capital / (current_price * (1/sqrt_price_lower - 1/sqrt_price_upper))
            elif current_price >= price_upper:
                self.liquidity_units = total_capital / (sqrt_price_upper - sqrt_price_lower)
            else:
                # Mixed case - simplified calculation
                self.liquidity_units = total_capital / (2 * sqrt_price)
            
            self.position_active = True
            self.cash = 0.0  # All capital is now in the position
        
        # Update position value
        self.liquidity_value = self.calculate_position_value(next_price)
        
        # Calculate reward (with hedging assumption)
        reward = trading_fee + lvr - gas_fee
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'trading_fee': trading_fee,
            'lvr': lvr,
            'gas_fee': gas_fee,
            'position_value': self.liquidity_value,
            'cash': self.cash
        }
        
        return self.get_state(), reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = self.initial_capital
        self.liquidity_value = 0.0
        self.liquidity_center = 0
        self.liquidity_width = 0
        self.liquidity_units = 0.0
        self.position_active = False
        
        return self.get_state()