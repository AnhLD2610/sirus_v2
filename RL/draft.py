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

class TechnicalIndicators:
    """Calculate technical indicators for state features"""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> np.ndarray:
        """Calculate all technical indicators from OHLCV data"""
        features = []
        
        # Basic price features
        features.append(df['open'].values[-1])  # Current open price
        features.append(df['high'].values[-1] / df['open'].values[-1])  # High/Open ratio
        features.append(df['low'].values[-1] / df['open'].values[-1])   # Low/Open ratio
        features.append(df['close'].values[-1] / df['open'].values[-1]) # Close/Open ratio
        features.append(df['volume'].values[-1])  # Trading volume
        
        # Technical indicators
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        # DEMA - Double Exponential Moving Average
        dema = talib.DEMA(close_prices, timeperiod=14)
        features.append(dema[-1] / df['open'].values[-1] if not np.isnan(dema[-1]) else 1.0)
        
        # SAR - Parabolic SAR
        sar = talib.SAR(high_prices, low_prices)
        features.append(sar[-1] / df['open'].values[-1] if not np.isnan(sar[-1]) else 1.0)
        
        # ADX - Average Directional Movement Index
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        features.append(adx[-1] if not np.isnan(adx[-1]) else 50.0)
        
        # APO - Absolute Price Oscillator
        apo = talib.APO(close_prices)
        features.append(apo[-1] if not np.isnan(apo[-1]) else 0.0)
        
        # AROON - Aroon Oscillator
        aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=14)
        aroon_osc = aroon_up[-1] - aroon_down[-1] if not (np.isnan(aroon_up[-1]) or np.isnan(aroon_down[-1])) else 0.0
        features.append(aroon_osc)
        
        # BOP - Balance of Power
        bop = talib.BOP(df['open'].values, high_prices, low_prices, close_prices)
        features.append(bop[-1] if not np.isnan(bop[-1]) else 0.0)
        
        # CCI - Commodity Channel Index
        cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        features.extend([cci[-1] if not np.isnan(cci[-1]) else 0.0, cci[-2] if len(cci) > 1 and not np.isnan(cci[-2]) else 0.0])
        
        # CMO - Chande Momentum Oscillator
        cmo = talib.CMO(close_prices, timeperiod=14)
        features.append(cmo[-1] if not np.isnan(cmo[-1]) else 0.0)
        
        # DX - Directional Movement Index
        dx = talib.DX(high_prices, low_prices, close_prices, timeperiod=14)
        features.append(dx[-1] if not np.isnan(dx[-1]) else 0.0)
        
        # MINUS_DM - Minus Directional Movement
        minus_dm = talib.MINUS_DM(high_prices, low_prices, timeperiod=14)
        features.append(minus_dm[-1] if not np.isnan(minus_dm[-1]) else 0.0)
        
        # MOM - Momentum
        mom = talib.MOM(close_prices, timeperiod=10)
        features.append(mom[-1] if not np.isnan(mom[-1]) else 0.0)
        
        # PLUS_DM - Plus Directional Movement
        plus_dm = talib.PLUS_DM(high_prices, low_prices, timeperiod=14)
        features.append(plus_dm[-1] if not np.isnan(plus_dm[-1]) else 0.0)
        
        # TRIX
        trix = talib.TRIX(close_prices, timeperiod=30)
        features.append(trix[-1] if not np.isnan(trix[-1]) else 0.0)
        
        # ULTOSC - Ultimate Oscillator
        ult_osc = talib.ULTOSC(high_prices, low_prices, close_prices)
        features.append(ult_osc[-1] if not np.isnan(ult_osc[-1]) else 50.0)
        
        # Stochastic indicators
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
        features.extend([
            slowk[-1] if not np.isnan(slowk[-1]) else 50.0,
            slowd[-1] if not np.isnan(slowd[-1]) else 50.0,
            slowk[-1] - slowd[-1] if not (np.isnan(slowk[-1]) or np.isnan(slowd[-1])) else 0.0
        ])
        
        # Fast Stochastic
        fastk, fastd = talib.STOCHF(high_prices, low_prices, close_prices)
        features.extend([
            fastk[-1] if not np.isnan(fastk[-1]) else 50.0,
            fastd[-1] if not np.isnan(fastd[-1]) else 50.0,
            fastk[-1] - fastd[-1] if not (np.isnan(fastk[-1]) or np.isnan(fastd[-1])) else 0.0
        ])
        
        # ATR - Average True Range
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        natr = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
        trange = talib.TRANGE(high_prices, low_prices, close_prices)
        features.extend([
            natr[-1] if not np.isnan(natr[-1]) else 1.0,
            trange[-1] if not np.isnan(trange[-1]) else 0.0
        ])
        
        # Hilbert Transform indicators
        ht_dcperiod = talib.HT_DCPERIOD(close_prices)
        ht_dcphase = talib.HT_DCPHASE(close_prices)
        features.extend([
            ht_dcperiod[-1] if not np.isnan(ht_dcperiod[-1]) else 15.0,
            ht_dcphase[-1] if not np.isnan(ht_dcphase[-1]) else 0.0
        ])
        
        return np.array(features, dtype=np.float32)

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

class DuelingDQN(nn.Module):
    """Dueling Double Deep Q-Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_units: List[int] = None):
        super(DuelingDQN, self).__init__()
        
        if hidden_units is None:
            hidden_units = Config.HIDDEN_UNITS
            
        # Shared feature layers
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Linear(input_dim, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(input_dim, action_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DRLAgent:
    """Deep Reinforcement Learning Agent for Uniswap V3 Liquidity Provision"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.q_network = DuelingDQN(state_dim, action_dim)
        self.target_network = DuelingDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=Config.LEARNING_RATE)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(Config.BUFFER_SIZE)
        
        # Training parameters
        self.epsilon = Config.EPSILON_START
        self.steps_done = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        """Update the network"""
        if len(self.memory) < Config.BATCH_SIZE:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(Config.BATCH_SIZE)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (Double DQN)
        next_actions = self.q_network(next_states).max(1)[1].detach()
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (Config.GAMMA * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), Config.GRADIENT_CLIP_NORM)
        self.optimizer.step()
        
        # Update target network
        self._soft_update()
        
        # Decay epsilon
        if self.epsilon > Config.EPSILON_END:
            self.epsilon *= Config.EPSILON_DECAY
    
    def _soft_update(self):
        """Soft update of target network"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                Config.TARGET_UPDATE_RATE * local_param.data + 
                (1.0 - Config.TARGET_UPDATE_RATE) * target_param.data
            )

def train_agent(agent: DRLAgent, env: UniswapV3Environment, episodes: int = 1000):
    """Train the DRL agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    return scores

def evaluate_agent(agent: DRLAgent, env: UniswapV3Environment) -> dict:
    """Evaluate trained agent"""
    state = env.reset()
    total_reward = 0
    total_trading_fee = 0
    total_gas_fee = 0
    total_lvr = 0
    actions_taken = []
    
    while True:
        action = agent.select_action(state, training=False)
        actions_taken.append(action)
        
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_trading_fee += info['trading_fee']
        total_gas_fee += info['gas_fee']
        total_lvr += info['lvr']
        
        state = next_state
        
        if done:
            break
    
    final_value = env.cash + env.liquidity_value
    relative_pnl = (final_value - env.initial_capital) / env.initial_capital
    
    return {
        'total_reward': total_reward,
        'relative_pnl': relative_pnl,
        'final_value': final_value,
        'trading_fee': total_trading_fee,
        'gas_fee': total_gas_fee,
        'lvr': total_lvr,
        'actions': actions_taken,
        'num_reallocations': sum(1 for a in actions_taken if a > 0)
    }

# Example usage and demonstration
def create_sample_data(n_hours: int = 2000) -> pd.DataFrame:
    """Create sample price data for demonstration"""
    np.random.seed(42)
    
    # Generate synthetic ETH/USDC price data
    initial_price = 2000.0
    prices = [initial_price]
    volumes = []
    
    for i in range(n_hours):
        # Geometric Brownian motion with some trend
        dt = 1/24  # 1 hour
        drift = 0.05 * dt  # 5% annual drift
        volatility = 0.8 * math.sqrt(dt)  # 80% annual volatility
        
        price_change = prices[-1] * (drift + volatility * np.random.normal())
        new_price = max(prices[-1] + price_change, 100.0)  # Minimum price floor
        prices.append(new_price)
        
        # Generate volume
        volume = np.random.lognormal(15, 1)  # Log-normal distribution for volume
        volumes.append(volume)
    
    # Create OHLC data
    data = []
    for i in range(1, len(prices)):
        open_price = prices[i-1]
        close_price = prices[i]
        
        # Generate high and low
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volumes[i-1]
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(2000)
    
    # Initialize environment
    print("Initializing environment...")
    env = UniswapV3Environment(data, initial_capital=1000.0)
    
    # Initialize agent
    state_dim = 32  # 28 technical indicators + 4 position features
    action_dim = Config.MAX_WIDTH + 1  # 0 (hold) + 1 to MAX_WIDTH (reallocate)
    
    print("Initializing DRL agent...")
    agent = DRLAgent(state_dim, action_dim)
    
    # Train agent
    print("Training agent...")
    scores = train_agent(agent, env, episodes=500)
    
    # Evaluate agent
    print("Evaluating agent...")
    results = evaluate_agent(agent, env)
    
    print("\n=== Evaluation Results ===")
    print(f"Relative P&L: {results['relative_pnl']:.4f}")
    print(f"Final Portfolio Value: ${results['final_value']:.2f}")
    print(f"Total Trading Fees: ${results['trading_fee']:.2f}")
    print(f"Total Gas Fees: ${results['gas_fee']:.2f}")
    print(f"Total LVR: ${results['lvr']:.2f}")
    print(f"Number of Reallocations: {results['num_reallocations']}")
    print(f"Average Action: {np.mean([a for a in results['actions'] if a > 0]):.2f}")
    
    print("\nImplementation complete! The agent has been trained to adaptively manage")
