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


class UniswapV3Environment:
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
        # Convert price to tick
        return int(math.log(price, 1.0001))
    
    def tick_to_price(self, tick: int) -> float:
        # Convert tick to price
        return 1.0001 ** tick
    

    # formula 1 
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
    
    # Use in formula 4 and 6 
    def calculate_position_value(self, price: float) -> float:
        # Calculate current value of liquidity position
        if not self.position_active or self.liquidity_units == 0:
            return 0.0
            
        tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
        tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
        
        x, y = self.calculate_reserves(price, tick_lower, tick_upper, self.liquidity_units)
        return price * x + y
    

    # formula 3 
    def calculate_trading_fee(self, price_from: float, price_to: float) -> float:
        # Calculate trading fees earned during price movement
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
    
    # formula 5 
    def calculate_lvr(self, price_from: float, price_to: float) -> float:
        # Calculate Loss-Versus-Rebalancing
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
        # Get current state representation

        cash_normalized = self.cash 
        center_tick_normalized = self.liquidity_center 
        width_normalized = self.liquidity_width 
        position_value_normalized = self.liquidity_value 
        
        state = np.concatenate([
            [cash_normalized, center_tick_normalized, width_normalized, position_value_normalized]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Execute one step in the environment
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
        # Reset environment to initial state
        self.current_step = 0
        self.cash = self.initial_capital
        self.liquidity_value = 0.0
        self.liquidity_center = 0
        self.liquidity_width = 0
        self.liquidity_units = 0.0
        self.position_active = False
        
        return self.get_state()