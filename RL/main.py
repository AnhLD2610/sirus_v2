import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import pickle
from datetime import datetime

# Import your environment (assuming it's in a file called uniswap_env.py)
# from uniswap_env import UniswapV3Environment, Config

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture that separately estimates state value V(s) 
    and advantage A(s,a), then combines them: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """
    def __init__(self, state_size: int, action_size: int, hidden_units: List[int] = [64, 64]):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature extraction layers
        layers = []
        input_size = state_size
        for hidden_size in hidden_units:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units[-1], hidden_units[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_units[-1] // 2, 1)
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units[-1], hidden_units[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_units[-1] // 2, action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        # Extract features
        features = self.feature_layers(state)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DuelingDQNAgent:
    """
    Dueling DQN Agent for Uniswap V3 liquidity management with the following features:
    - Dueling network architecture
    - Experience replay
    - Target network with soft updates
    - Epsilon-greedy exploration with decay
    - Gradient clipping
    """
    
    def __init__(self, state_size: int, action_size: int, config=None):
        if config is None:
            from uniswap_env import Config
            config = Config
            
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DuelingDQN(state_size, action_size, config.HIDDEN_UNITS).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size, config.HIDDEN_UNITS).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        
        # Replay buffer
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        
        # Exploration parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        
        # Training parameters
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.target_update_rate = config.TARGET_UPDATE_RATE
        self.gradient_clip_norm = config.GRADIENT_CLIP_NORM
        
        # Metrics tracking
        self.training_losses = []
        self.episode_rewards = []
        self.epsilon_history = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network (Double DQN style)
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip_norm)
        
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Track metrics
        self.training_losses.append(loss.item())
        self.epsilon_history.append(self.epsilon)
    
    def _soft_update(self):
        """Soft update target network parameters"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.target_update_rate * local_param.data + 
                (1.0 - self.target_update_rate) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save model and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards,
            'epsilon_history': self.epsilon_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_losses = checkpoint['training_losses']
        self.episode_rewards = checkpoint['episode_rewards']
        self.epsilon_history = checkpoint['epsilon_history']
        print(f"Model loaded from {filepath}")

class UniswapV3Trainer:
    """Training manager for the Dueling DQN agent"""
    
    def __init__(self, env, agent, validation_env=None):
        self.env = env
        self.agent = agent
        self.validation_env = validation_env
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.validation_rewards = []
        self.training_start_time = None
    
    def train(self, num_episodes: int, save_interval: int = 100, validate_interval: int = 50):
        """Train the agent for specified number of episodes"""
        self.training_start_time = datetime.now()
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Action space size: {self.agent.action_size}")
        print(f"State space size: {self.agent.state_size}")
        
        best_validation_reward = float('-inf')
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            while True:
                # Select action
                action = self.agent.act(state, training=True)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train
                self.agent.replay()
                
                # Update state
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)
            self.agent.episode_rewards.append(episode_reward)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}, "
                      f"Buffer Size: {len(self.agent.memory)}")
            
            # Validation
            if self.validation_env and episode % validate_interval == 0 and episode > 0:
                val_reward = self.validate()
                self.validation_rewards.append(val_reward)
                print(f"Validation Reward: {val_reward:.2f}")
                
                # Save best model
                if val_reward > best_validation_reward:
                    best_validation_reward = val_reward
                    self.agent.save(f"best_model_episode_{episode}.pth")
            
            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                self.agent.save(f"checkpoint_episode_{episode}.pth")
        
        training_time = datetime.now() - self.training_start_time
        print(f"Training completed in {training_time}")
        
        # Final save
        self.agent.save("final_model.pth")
    
    def validate(self, num_episodes: int = 5) -> float:
        """Validate the agent's performance"""
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.validation_env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.act(state, training=False)
                next_state, reward, done, _ = self.validation_env.step(action)
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def plot_metrics(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Moving average of rewards
        if len(self.episode_rewards) >= 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Moving Average Reward (10 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
        
        # Training loss
        if self.agent.training_losses:
            axes[1, 0].plot(self.agent.training_losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Epsilon decay
        if self.agent.epsilon_history:
            axes[1, 1].plot(self.agent.epsilon_history)
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data for training"""
    # Ensure we have the required columns
    required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    
    # Convert timestamp if needed
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Sort by timestamp
    df = df.sort_values('open_time').reset_index(drop=True)
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def main():
    """Main training script"""
    df = pd.read_csv('ETH-USDT.csv')
    df = prepare_data(df)
    
    # For demonstration, create sample data
    # np.random.seed(42)
    # n_samples = 1000
    # df = pd.DataFrame({
    #     'open_time': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
    #     'open': 300 + np.cumsum(np.random.randn(n_samples) * 0.1),
    #     'high': 0,
    #     'low': 0,
    #     'close': 0,
    #     'volume': np.random.exponential(1000, n_samples)
    # })
    # df['high'] = df['open'] + np.random.exponential(0.5, n_samples)
    # df['low'] = df['open'] - np.random.exponential(0.5, n_samples)
    # df['close'] = df['open'] + np.random.randn(n_samples) * 0.2
    
    # Split data
    train_size = int(0.8 * len(df))
    train_df = df[:train_size].copy()
    val_df = df[train_size:].copy()
    
    # Create environments
    from uniswap_env import UniswapV3Environment, Config
    train_env = UniswapV3Environment(train_df, initial_capital=10000.0)
    val_env = UniswapV3Environment(val_df, initial_capital=10000.0)
    
    # Create agent
    state_size = len(train_env.get_state())
    action_size = Config.MAX_WIDTH + 1  # +1 for hold action
    agent = DuelingDQNAgent(state_size, action_size)
    
    # Create trainer
    trainer = UniswapV3Trainer(train_env, agent, val_env)
    
    # Train
    trainer.train(num_episodes=1000, save_interval=100, validate_interval=50)
    
    # Plot results
    trainer.plot_metrics()
    
    print("Training completed!")

if __name__ == "__main__":
    main()