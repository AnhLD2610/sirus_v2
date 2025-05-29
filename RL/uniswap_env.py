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

class Config:
    # Environment parameters
    TICK_SPACING = 60  # 0.3% fee pools
    MAX_WIDTH = 50     # Maximum liquidity interval width
    GAS_FEE = 1  # Gas fee  
    
    # DRL parameters 
    HIDDEN_UNITS = [1024, 1024]
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    BUFFER_SIZE = int(1e4)
    GAMMA = 0.99
    TARGET_UPDATE_RATE = 0.005
    GRADIENT_CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TRAIN_FREQUENCY = 1


class UniswapV3Environment:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0, fee_tier: float = 0.003, max_episode_steps: int = 1000):
        data.columns = data.columns.str.replace('**', '')
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.fee_tier = fee_tier
        self.tick_spacing = Config.TICK_SPACING
        self.max_episode_steps = max_episode_steps
        
        # State variables
        self.current_step = 0
        self.episode_start_step = 0
        self.cash = initial_capital
        self.liquidity_value = 0.0
        self.liquidity_center = 0
        self.liquidity_width = 0
        self.liquidity_units = 0.0
        
        # Position tracking
        self.position_active = False
        
    def price_to_tick(self, price: float) -> int:
        return int(math.log(price, 1.0001))
    
    def tick_to_price(self, tick: int) -> float:
        return 1.0001 ** tick

    def calculate_reserves(self, price: float, tick_lower: int, tick_upper: int, liquidity: float) -> Tuple[float, float]:
        price_lower = self.tick_to_price(tick_lower)
        price_upper = self.tick_to_price(tick_upper)
        sqrt_price = math.sqrt(price)
        sqrt_price_lower = math.sqrt(price_lower)
        sqrt_price_upper = math.sqrt(price_upper)
        
        if price <= price_lower:
            x = liquidity * (1/sqrt_price_lower - 1/sqrt_price_upper)
            y = 0
        elif price >= price_upper:
            x = 0
            y = liquidity * (sqrt_price_upper - sqrt_price_lower)
        else:
            x = liquidity * (1/sqrt_price - 1/sqrt_price_upper)
            y = liquidity * (sqrt_price - sqrt_price_lower)
            
        return x, y
    
    def calculate_position_value(self, price: float) -> float:
        if not self.position_active or self.liquidity_units == 0:
            return 0.0
            
        tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
        tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
        
        x, y = self.calculate_reserves(price, tick_lower, tick_upper, self.liquidity_units)
        return price * x + y

    def calculate_trading_fee(self, price_from: float, price_to: float) -> float:
        if not self.position_active or self.liquidity_units == 0:
            return 0.0
            
        tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
        tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
        price_lower = self.tick_to_price(tick_lower)
        price_upper = self.tick_to_price(tick_upper)
        
        if price_from <= price_lower and price_to <= price_lower:
            return 0.0
        if price_from >= price_upper and price_to >= price_upper:
            return 0.0
            
        sqrt_price_from = max(math.sqrt(price_lower), min(math.sqrt(price_upper), math.sqrt(price_from)))
        sqrt_price_to = max(math.sqrt(price_lower), min(math.sqrt(price_upper), math.sqrt(price_to)))
        
        fee = 0.0
        if price_to > price_from:
            fee = (self.fee_tier / (1 - self.fee_tier)) * self.liquidity_units * (sqrt_price_to - sqrt_price_from)
        else:
            fee = (self.fee_tier / (1 - self.fee_tier)) * self.liquidity_units * (1/sqrt_price_from - 1/sqrt_price_to) * price_from
            
        return max(0.0, fee)
    
    def calculate_lvr(self, price_from: float, price_to: float) -> float:
        if not self.position_active or self.liquidity_units == 0:
            return 0.0
            
        tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
        tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
        
        value_from = self.calculate_position_value(price_from)
        value_to = self.calculate_position_value(price_to)
        
        x_from, _ = self.calculate_reserves(price_from, tick_lower, tick_upper, self.liquidity_units)
        
        lvr = (value_to - value_from) - x_from * (price_to - price_from)
        
        return lvr
    
    def get_state(self) -> np.ndarray:
        current_row = self.data.iloc[self.current_step]
        
        cash_normalized = self.cash 
        center_tick_normalized = self.liquidity_center 
        width_normalized = self.liquidity_width 
        position_value_normalized = self.liquidity_value 
        
        current_price = current_row['close']
        price_normalized = current_price
        
        volume = current_row['volume']
        volume_normalized = volume
        
        quote_asset_volume = current_row.get('quote_asset_volume', 0)
        number_of_trades = current_row.get('number_of_trades', 0)
        taker_buy_base_volume = current_row.get('taker_buy_base_asset_volume', 0)
        taker_buy_quote_volume = current_row.get('taker_buy_quote_asset_volume', 0)
    
        state = np.array([
            cash_normalized,
            center_tick_normalized, 
            width_normalized,
            position_value_normalized,
            price_normalized,
            volume_normalized,
            quote_asset_volume,      
            number_of_trades,            
            taker_buy_base_volume,         
            taker_buy_quote_volume         
        ])
        
        return state.astype(np.float32)
    
    def calculate_liquidity(self, current_price, price_lower, price_upper, capital):
        sqrt_p = math.sqrt(current_price)
        sqrt_pa = math.sqrt(price_lower)
        sqrt_pb = math.sqrt(price_upper)
        
        if current_price <= price_lower:
            delta_x = capital / current_price
            L = delta_x * (sqrt_pa * sqrt_pb) / (sqrt_pb - sqrt_pa)
        elif current_price >= price_upper:
            delta_y = capital
            L = delta_y / (sqrt_pb - sqrt_pa)
        else:
            numerator = capital
            denominator = (
                (sqrt_pb - sqrt_p) * sqrt_p / sqrt_pb +
                (sqrt_p - sqrt_pa)
            )
            L = numerator / denominator
        return L

    def step(self, action: int):
        if self.current_step >= len(self.data) - 1:
            return self.get_state(), 0.0, True, {}

        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']

        # STEP 1: Calculate value BEFORE any changes
        prev_total_value = self.cash + self.calculate_position_value(current_price)
        
        # STEP 2: Calculate trading fees from existing position
        trading_fee = 0.0
        if self.position_active:
            trading_fee = self.calculate_trading_fee(current_price, next_price)

        # STEP 3: Handle position changes
        gas_fee = 0.0
        if action != 0: 
            gas_fee = Config.GAS_FEE

            # Close existing position and open new one
            capital_for_new_lp = self.cash
            if self.position_active:
                capital_for_new_lp += self.calculate_position_value(current_price)
                # Reset position
                self.liquidity_value = 0.0 
                self.liquidity_units = 0.0

            # Create new position
            current_tick = self.price_to_tick(current_price)
            self.liquidity_center = self.tick_spacing * round(current_tick / self.tick_spacing)
            self.liquidity_width = action 

            tick_lower = self.liquidity_center - self.liquidity_width * self.tick_spacing
            tick_upper = self.liquidity_center + self.liquidity_width * self.tick_spacing
            price_lower = self.tick_to_price(tick_lower)
            price_upper = self.tick_to_price(tick_upper)

            if price_lower >= price_upper:
                self.liquidity_units = 0.0
            else:
                self.liquidity_units = self.calculate_liquidity(
                    current_price, price_lower, price_upper, capital_for_new_lp
                )

            if self.liquidity_units > 0:
                self.position_active = True
                self.liquidity_value = self.calculate_position_value(current_price)
                self.cash = 0.0
            else:
                self.position_active = False
                self.cash = capital_for_new_lp
                self.liquidity_value = 0.0

        new_total_value = self.cash + self.calculate_position_value(next_price)
        
        value_change = new_total_value - prev_total_value
        # print(value_change)
        # reward = trading_fee + value_change - gas_fee

        lvr = self.calculate_lvr(current_price, next_price)
        reward = trading_fee + lvr - gas_fee

        # print(lvr)
        # print(reward)
        # print(f"fee = {trading_fee:.10e}   lvr = {value_change:.10e}   gas = {gas_fee:.10f}")

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        info = {
            'trading_fee': trading_fee,
            'value_change': value_change,
            'gas_fee': gas_fee,
            'prev_total_value': prev_total_value,
            'new_total_value': new_total_value,
            'position_value': self.calculate_position_value(next_price),
            'cash': self.cash
        }

        return self.get_state(), reward, done, info
    
        # print(f"fee = {trading_fee:.10e}   lvr = {new_total_value - prev_total_value:.10e}   gas = {gas_fee:.10f}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.cash = self.initial_capital
        self.liquidity_value = 0.0
        self.liquidity_center = 0
        self.liquidity_width = 0
        self.liquidity_units = 0.0
        self.position_active = False
        
        return self.get_state()