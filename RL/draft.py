import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment:
    """
    Trading environment for reinforcement learning
    """

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001, window_size: int = 20):
        """
        Initialize trading environment

        Args:
            data: DataFrame with OHLCV data
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as percentage
            window_size: Number of previous periods to include in state
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.total_profit = 0
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step < self.window_size:
            return np.zeros(self.window_size * 8 + 3)

        # Price and volume features
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        features = []

        # OHLCV normalized data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            features.extend(self.data[col].iloc[start_idx:end_idx].values)

        current_data = self.data.iloc[self.current_step - 1]
        features = current_data

        # Portfolio state
        current_price = self.data['close'].iloc[self.current_step - 1]
        portfolio_value = self.balance + self.shares * current_price

        features.extend([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares * current_price / self.initial_balance,  # Normalized position value
            portfolio_value / self.initial_balance  # Normalized total value
        ])

        return np.array(features, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return next state, reward, done flag, and info

        Actions:
        0: Hold
        1: Buy
        2: Sell
        """
        if self.current_step >= len(self.data):
            return self._get_state(), 0, True, {}

        current_price = self.data['close'].iloc[self.current_step]
        previous_price = self.data['close'].iloc[self.current_step - 1]

        # Calculate reward based on previous action's performance
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = self.balance // (current_price * (1 + self.transaction_cost))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares += shares_to_buy
                self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # Sell
            if self.shares > 0:
                revenue = self.shares * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trades.append(('SELL', self.current_step, current_price, self.shares))
                self.shares = 0

        # Calculate portfolio value and reward
        portfolio_value = self.balance + self.shares * current_price
        self.portfolio_values.append(portfolio_value)

        # Reward calculation
        portfolio_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        market_return = (current_price - previous_price) / previous_price

        # Reward for outperforming market
        reward = portfolio_return - market_return

        # Penalty for large drawdowns
        if len(self.portfolio_values) > 1:
            max_value = max(self.portfolio_values)
            current_drawdown = (max_value - portfolio_value) / max_value
            reward -= current_drawdown * 0.1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'current_price': current_price
        }

        return self._get_state(), reward, done, info


class DQNAgent:
    """
    Deep Q-Network agent for trading
    """

    def __init__(self, state_size: int, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)

        # Simple neural network weights (since we can't use tensorflow/pytorch)
        self.weights = self._initialize_weights()

    def _initialize_weights(self):
        """Initialize neural network weights"""
        # Simple 3-layer network
        weights = {}

        # Layer 1: state_size -> 64
        weights['W1'] = np.random.randn(self.state_size, 64) * 0.1
        weights['b1'] = np.zeros(64)

        # Layer 2: 64 -> 32
        weights['W2'] = np.random.randn(64, 32) * 0.1
        weights['b2'] = np.zeros(32)

        # Layer 3: 32 -> action_size
        weights['W3'] = np.random.randn(32, self.action_size) * 0.1
        weights['b3'] = np.zeros(self.action_size)

        return weights

    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def _forward(self, state):
        """Forward pass through network"""
        # Layer 1
        z1 = np.dot(state, self.weights['W1']) + self.weights['b1']
        a1 = self._relu(z1)

        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self._relu(z2)

        # Layer 3 (output)
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']

        return z3, (z1, a1, z2, a2, z3)

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values, _ = self._forward(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_q_values, _ = self._forward(next_state.reshape(1, -1))
                target = reward + 0.95 * np.amax(next_q_values[0])

            current_q_values, cache = self._forward(state.reshape(1, -1))
            target_q_values = current_q_values.copy()
            target_q_values[0][action] = target

            # Simple gradient descent update
            self._backward(state.reshape(1, -1), target_q_values, cache)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _backward(self, state, target_q_values, cache):
        """Backward pass (simplified gradient descent)"""
        z1, a1, z2, a2, z3 = cache

        # Output layer error
        dz3 = z3 - target_q_values
        dW3 = np.dot(a2.T, dz3) * self.learning_rate
        db3 = np.sum(dz3, axis=0) * self.learning_rate

        # Hidden layer 2 error
        da2 = np.dot(dz3, self.weights['W3'].T)
        dz2 = da2 * (z2 > 0)  # ReLU derivative
        dW2 = np.dot(a1.T, dz2) * self.learning_rate
        db2 = np.sum(dz2, axis=0) * self.learning_rate

        # Hidden layer 1 error
        da1 = np.dot(dz2, self.weights['W2'].T)
        dz1 = da1 * (z1 > 0)  # ReLU derivative
        dW1 = np.dot(state.T, dz1) * self.learning_rate
        db1 = np.sum(dz1, axis=0) * self.learning_rate

        # Update weights
        self.weights['W3'] -= dW3
        self.weights['b3'] -= db3
        self.weights['W2'] -= dW2
        self.weights['b2'] -= db2
        self.weights['W1'] -= dW1
        self.weights['b1'] -= db1


def load_data_from_string(data_string: str) -> pd.DataFrame:
    """Load trading data from string format"""
    lines = data_string.strip().split('\n')

    # Skip header lines and empty lines
    data_lines = [line for line in lines if line.strip() and not line.startswith('open_time')]

    data = []
    for line in data_lines:
        values = line.split()
        if len(values) >= 6:  # Ensure we have enough values
            try:
                row = {
                    'open_time': float(values[0]),
                    'open': float(values[1]),
                    'high': float(values[2]),
                    'low': float(values[3]),
                    'close': float(values[4]),
                    'volume': float(values[5])
                }
                data.append(row)
            except ValueError:
                continue  # Skip malformed lines

    return pd.DataFrame(data)


def train_agent(env: TradingEnvironment, agent: DQNAgent, episodes: int = 100):
    """Train the DQN agent"""
    scores = []
    portfolio_values = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                portfolio_values.append(info['portfolio_value'])
                break

        scores.append(total_reward)
        agent.replay()

        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_portfolio = np.mean(portfolio_values[-10:])
            print(f"Episode {episode}, Avg Score: {avg_score:.4f}, "
                  f"Avg Portfolio Value: ${avg_portfolio:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")

    return scores, portfolio_values


def backtest_strategy(env: TradingEnvironment, agent: DQNAgent):
    """Backtest the trained strategy"""
    state = env.reset()
    actions_taken = []

    while True:
        # Use greedy policy (no exploration)
        old_epsilon = agent.epsilon
        agent.epsilon = 0
        action = agent.act(state)
        agent.epsilon = old_epsilon

        actions_taken.append(action)
        next_state, reward, done, info = env.step(action)
        state = next_state

        if done:
            break

    return actions_taken, env.portfolio_values, env.trades


def plot_results(portfolio_values: List[float], prices: np.ndarray, trades: List[Tuple]):
    """Plot trading results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value over time
    ax1.plot(portfolio_values, label='Portfolio Value', color='blue')
    ax1.axhline(y=portfolio_values[0], color='red', linestyle='--', label='Initial Value')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True)

    # Price with buy/sell signals
    ax2.plot(prices, label='Price', color='black', alpha=0.7)

    # Mark trades
    for trade in trades:
        trade_type, step, price, shares = trade
        color = 'green' if trade_type == 'BUY' else 'red'
        marker = '^' if trade_type == 'BUY' else 'v'
        ax2.scatter(step, price, color=color, marker=marker, s=100,
                   label=f'{trade_type}' if trade == trades[0] or
                   (trade_type == 'SELL' and trades[0][0] == 'BUY') else "")

    ax2.set_title('Price with Trading Signals')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Sample data string (replace with your actual data)
    sample_data = """
    1.5E+12 301.13 301.13 300 301.13 3.82951 1.5E+12 1152.713 12 3.51562 1058.546 0
    1.5E+12 300 301.13 298 298 1.97216 1.5E+12 592.0527 10 1.9683 590.9024 0
    1.5E+12 298 298 298 298 0 1.5E+12 0 0 0 0 0
    1.5E+12 298 299.05 298 299.05 12.88486 1.5E+12 3840.085 4 3.28872 980.435 0
    1.5E+12 299.05 300.1 299.05 300.1 6.58304 1.5E+12 1970.701 8 0.51388 153.874 0
    1.5E+12 299.4 300.8 299.39 299.39 11.97275 1.5E+12 3586.132 14 8.31452 2490.858 0
    1.5E+12 299.39 299.39 299.39 299.39 20.79097 1.5E+12 6224.609 14 20.79097 6224.609 0
    1.5E+12 299.39 300.79 299.39 299.6 27.59262 1.5E+12 8262.628 8 3.03842 911.3458 0
    1.5E+12 299.6 299.6 299.6 299.6 4.5522 1.5E+12 1363.839 17 4.5522 1363.839 0
    1.5E+12 299.6 300.8 299.6 300.79 5.23649 1.5E+12 1574.183 12 4.76497 1432.916 0
    1.5E+12 300.79 301.13 300.79 301.13 15.94388 1.5E+12 4799.272 20 15.50588 4667.526 0
    1.5E+12 301.13 302.57 301.13 301.61 14.31029 1.5E+12 4318.59 10 14.31029 4318.59 0
    """

    # Load data
    data = load_data_from_string(sample_data)
    print(f"Loaded {len(data)} data points")
    print(data.head())

    # Create environment and agent
    env = TradingEnvironment(data, initial_balance=10000)
    state_size = len(env._get_state())
    agent = DQNAgent(state_size)

    print(f"State size: {state_size}")
    print("Starting training...")

    # Train agent
    scores, portfolio_values = train_agent(env, agent, episodes=50)

    # Backtest
    print("\nBacktesting...")
    actions, final_portfolio_values, trades = backtest_strategy(env, agent)

    # Results
    initial_value = final_portfolio_values[0]
    final_value = final_portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100

    print(f"\nBacktest Results:")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")

    # Plot results
    plot_results(final_portfolio_values, data['close'].values, trades)