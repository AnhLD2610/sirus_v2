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
from datetime import datetime

from uniswap_env import UniswapV3Environment, Config

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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        features = self.feature_layers(state)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class ReplayBuffer:    
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
    
    def __init__(self, state_size: int, action_size: int, config=None):
        if config is None:
            config = Config
            
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
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
        self.train_frequency = config.TRAIN_FREQUENCY
        
        # Training step counter
        self.step_count = 0
        
        # Metrics tracking
        self.training_losses = []
        self.episode_rewards = []
        self.epsilon_history = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        # Select action using epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        # Train the network on a batch of experiences
        self.step_count += 1
        
        if (self.step_count % self.train_frequency != 0 or 
            len(self.memory) < self.batch_size):
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
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
        
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.training_losses.append(loss.item())
        self.epsilon_history.append(self.epsilon)
    
    def _soft_update(self):
        # Soft update target network parameters
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.target_update_rate * local_param.data + 
                (1.0 - self.target_update_rate) * target_param.data
            )
    
    def save(self, filepath: str):
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
        self.validation_episodes = []  # Track which episodes had validation
        self.training_start_time = None
    
    def train(self, num_episodes: int, save_interval: int = 100, validate_interval: int = 50):

        import os
        os.makedirs("checkpoints", exist_ok=True)

        self.training_start_time = datetime.now()
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Action space size: {self.agent.action_size}")
        print(f"State space size: {self.agent.state_size}")
        print(f"Training frequency: every {self.agent.train_frequency} steps")
        print(f"Batch size: {self.agent.batch_size}")
        print("-" * 50)
        
        best_validation_reward = float('-inf')
        
        for episode in range(num_episodes):
            if episode % 10 == 0:
                elapsed = datetime.now() - self.training_start_time
                print(f"Episode {episode}/{num_episodes} - Elapsed: {elapsed}")
                
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
            
            # metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)
            self.agent.episode_rewards.append(episode_reward)
            
            # Print progress 
            if episode % 1 == 0 and episode > 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                elapsed = datetime.now() - self.training_start_time
                print(f"\n=== Episode {episode} Summary ===")
                print(f"Avg Reward (50 eps): {avg_reward:.2f}")
                print(f"Avg Episode Length: {avg_length:.1f}")
                print(f"Epsilon: {self.agent.epsilon:.3f}")
                print(f"Buffer Size: {len(self.agent.memory)}")
                print(f"Training Steps: {self.agent.step_count}")
                print(f"Elapsed Time: {elapsed}")
                print("-" * 30)
            
            # Validation - MODIFIED TO SAVE VALUES
            if self.validation_env and episode % validate_interval == 0 and episode > 0:
                val_reward = self.validate()
                self.validation_rewards.append(val_reward)
                self.validation_episodes.append(episode)  # Track which episode this validation was for
                print(f"Validation Reward: {val_reward:.2f}")
                
                # Save best model
                if val_reward > best_validation_reward:
                    best_validation_reward = val_reward
                    self.agent.save(f"checkpoints/best_model_episode_{episode}.pth")
            
            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                self.agent.save(f"checkpoints/checkpoint_episode_{episode}.pth")
        
        training_time = datetime.now() - self.training_start_time
        print(f"\nTraining completed in {training_time}")
        print(f"Final buffer size: {len(self.agent.memory)}")
        print(f"Total training steps: {self.agent.step_count}")
        
        self.agent.save("checkpoints/final_model.pth")
        
        # PLOT VALIDATION REWARDS
        self.plot_validation_results()
    
    def plot_validation_results(self):
        """Plot validation rewards over training episodes"""
        if not self.validation_rewards:
            print("No validation data to plot.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Validation Rewards
        ax1.plot(self.validation_episodes, self.validation_rewards, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Validation Reward')
        ax1.set_title('Validation Performance Over Training')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        ax1.legend()
        
        # Add statistics
        max_val = max(self.validation_rewards)
        min_val = min(self.validation_rewards)
        avg_val = np.mean(self.validation_rewards)
        ax1.text(0.02, 0.98, f'Max: {max_val:.2f}\nMin: {min_val:.2f}\nAvg: {avg_val:.2f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Training Rewards (if available)
        if self.episode_rewards:
            # Moving average for smoother visualization
            window_size = min(50, len(self.episode_rewards) // 10)
            if window_size > 1:
                moving_avg = pd.Series(self.episode_rewards).rolling(window=window_size).mean()
                ax2.plot(range(len(self.episode_rewards)), self.episode_rewards, alpha=0.3, color='gray', label='Episode Rewards')
                ax2.plot(range(len(moving_avg)), moving_avg, 'g-', linewidth=2, label=f'Moving Average ({window_size} eps)')
            else:
                ax2.plot(range(len(self.episode_rewards)), self.episode_rewards, 'g-', linewidth=1, label='Episode Rewards')
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Training Reward')
            ax2.set_title('Training Performance Over Episodes')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"validation_results_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Validation plot saved as: {plot_filename}")
        
        plt.show()

    def validate(self, num_episodes: int = 5) -> float:
        metrics = []
        for _ in range(num_episodes):
            # Reset to get initial_total (cash + position)
            initial_state = self.validation_env.reset()
            init_cash = self.validation_env.cash
            init_pos = self.validation_env.calculate_position_value(
                self.validation_env.data.iloc[self.validation_env.current_step]['close']
            )
            initial_total = init_cash + init_pos

            # run 1 episode
            state = initial_state
            done = False
            while not done:
                action = self.agent.act(state, training=False)
                next_state, _, done, _ = self.validation_env.step(action)
                state = next_state

            # get final value
            final_cash = self.validation_env.cash
            final_pos = self.validation_env.calculate_position_value(
                self.validation_env.data.iloc[self.validation_env.current_step]['close']
            )
            final_total = final_cash + final_pos

            # metric
            profit = final_total - initial_total
            metrics.append(profit)  

        # mean
        return float(np.mean(metrics))

def prepare_data(df: pd.DataFrame, max_rows: int = 100000) -> pd.DataFrame:
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    df = df.sort_values('open_time').reset_index(drop=True)
    
    if len(df) > max_rows:
        df = df.head(max_rows)
        print(f"Data limited to {max_rows} rows (original: {len(df)} rows)")
    
    df = df.ffill().bfill()  
    
    return df




def main():
    df = pd.read_csv('ETH-USDT.csv')
    df = prepare_data(df)
    
#     train_size = int(0.8 * len(df))
#     train_df = df[:train_size].copy()
#     val_df = df[train_size:].copy()
    
#     # Create environments
#     train_env = UniswapV3Environment(train_df, initial_capital=10000.0)
#     val_env = UniswapV3Environment(val_df, initial_capital=10000.0)
    
#     # Create agent
#     state_size = len(train_env.get_state())
#     action_size = Config.MAX_WIDTH + 1  # +1 for hold action
#     agent = DuelingDQNAgent(state_size, action_size)
    
#     # Create trainer
#     trainer = UniswapV3Trainer(train_env, agent, val_env)
    
#     # Train
#     trainer.train(num_episodes=10, save_interval=1, validate_interval=1)
    

# if __name__ == "__main__":
#     main()