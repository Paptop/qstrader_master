import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from examples.alpha_models.ExperienceBuffer import ExperienceBuffer
from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.statistics.tearsheet import TearsheetStatistics
import qstrader.statistics.performance as perf
import os

class QLearningAlphaModel(AlphaModel):
    def __init__(
        self,
        signals,
        universe,
        data_handler,
        performance_stats=None,
        learning_rate=0.001,
        discount_factor=0.9,
        exploration_rate=1.0,  # Start with high exploration
        min_exploration_rate=0.01,  # Minimum exploration rate
        decay_rate=0.99,  # Exponential decay rate
        state_size=3,
        action_size=2,
        eval_mode=False, # Evaluation mode flag
        benchmark_curve=None,
        buffer_capacity=10000,
        batch_size=64
    ):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.decay_rate = decay_rate
        self.state_size = state_size
        self.action_size = action_size
        self.eval_mode = eval_mode  # Initialize evaluation mode
        self.benchmark_curve = benchmark_curve

        # Primary Q-network
        self.q_network = self._build_network()
        # Target Q-network
        self.target_q_network = self._build_network()
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize with same weights
        self.target_q_network.eval()  # Target network is in evaluation mode

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.weights = None
        self.previous_state = None
        self.previous_action = None
        self.action_mapping = {
            0: "long",
            1: "short",
            #2: "neutral"
        }
        self.isInvested = False
        self.counter = 0

        self.experience_buffer = ExperienceBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.epoch_number = 0
        self.performance_stats = performance_stats
        self.consecutive_long_actions = 0  # Counter for consecutive "long" actions
        self.max_consecutive_long = 16  # Threshold for applying a penalty
        self.start_date = None

    def reset_memory(self):
        self.experience_buffer = ExperienceBuffer(10000)

    def initialize(self, epoch_number):
        """
        Set the epoch number for the model.

        Parameters
        ----------
        epoch_number : int
            The current epoch number.
        """
        self.epoch_number = epoch_number
        self.consecutive_long_actions = 0

    def _build_network(self):
        """
        Build a simple feedforward neural network for Q-learning.
        """
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def update_target_network(self):
        """
        Update the target Q-network with the weights of the primary Q-network.
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        print("Target network updated.")

    def update_target_network_soft(self, tau=0.01):
        """
        Soft update the target Q-network with the weights of the primary Q-network.

        Parameters
        ----------
        tau : float
            The interpolation factor for the soft update. A value of 0.01 means
            1% of the primary Q-network's weights are blended into the target Q-network.
        """
        for target_param, q_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
        #print("Target network updated with soft update.")

    def get_description(self):
        return {
            "name": "QLearningAlphaModel",
            "description": "Alpha model using Q-learning to adjust weights dynamically",
            "parameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate
            }
        }

    def _get_state(self, dt):
        """
        Define the state based on market conditions and normalize it.
        """
        # Define signals

        # Define signals
        short_sma = 8
        long_sma = 20

        short_wma = 12
        long_wma = 32

        long_ema = 10
        short_ema = 40

        kama_period = 365

        rsi_period_long = 30
        rsi_period_short = 13

        assets = self.universe.get_assets(dt)
        short_ema_sp = self.signals['ema'](assets[0], short_ema)
        long_ema_sp = self.signals['ema'](assets[0], long_ema)

        short_wma_sp = self.signals['wma'](assets[0], short_wma)
        long_wma_sp = self.signals['wma'](assets[0], long_wma)

        short_sma_sp = self.signals['sma'](assets[0], short_sma)
        long_sma_sp = self.signals['sma'](assets[0], long_sma)

        kama = self.signals['kama'](assets[0], kama_period)

        rsi_long = self.signals['rsi'](assets[0], int(rsi_period_long))
        rsi_short = self.signals['rsi'](assets[0], int(rsi_period_short))

        close_price_sp = self.signals['cur_price'](assets[0], 1)[0]
        close_price_b = self.signals['cur_price'](assets[1], 1)[0]

        rs_sma_sp = short_sma_sp / long_sma_sp
        rs_ema_sp = short_ema_sp / long_ema_sp
        rs_wma_sp = short_wma_sp / long_wma_sp
        rs_kama = kama / close_price_sp
        rs_current =  close_price_b / close_price_sp
        rs_rsi = rsi_short / rsi_long

        short_ema_b = self.signals['ema'](assets[1], short_ema)
        long_ema_b = self.signals['ema'](assets[1], long_ema)

        rs_long = long_ema_sp / long_ema_b
        rs_short = short_ema_sp / short_ema_b
        rs_current = close_price_sp / close_price_b
        # Normalize the state values
        return np.array([
            #rs_sma_sp,
            #rs_ema_sp,
            #rs_wma_sp,
            #rs_kama,
            #rs_current,
            #rs_rsi,
            rs_long,
            rs_short,
            rs_current,
        ], dtype=np.float32)

    def _choose_action(self, state):
        """
        Choose an action based on the Q-network and exploration rate.
        """
        if np.random.rand() < self.exploration_rate:
            # Explore: choose a random action
            action = np.random.choice([0, 1])
            print("Exploring: ", action)
        else:
            # Exploit: choose the best action based on the Q-network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor).detach().numpy()
            action = np.argmax(q_values)
            print("Exploiting: ", action)

        # Decay the exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.decay_rate
        )

        return action

    def set_eval_mode(self, eval_mode):
        """
        Set the evaluation mode for the model.

        Parameters
        ----------
        eval_mode : bool
            If True, the model will not update the Q-network.
        """
        self.eval_mode = eval_mode
        if eval_mode:
            self.q_network.eval()  # Set the network to evaluation mode
        else:
            self.q_network.train()  # Set the network to training mode

    def _update_q_network(self, reward, current_state, action):
        """
        Update the Q-network using the Double Deep Q-Learning formula.
        """
        if self.eval_mode:
            # Skip updating the Q-network in evaluation mode
            return
        
        # Add the current transition to the experience buffer
        done = False  # Set to True if the episode ends (can be updated based on your logic)
        if self.previous_state is not None and self.previous_action is not None:
            self.experience_buffer.add(self.previous_state, self.previous_action, reward, current_state, done)

        # Only update the Q-network if the buffer has enough samples
        if self.experience_buffer.size() < self.batch_size:
            return
        
        # Sample a batch of experiences from the buffer
        batch = self.experience_buffer.sample(self.batch_size)

        # Loop over the sampled experiences
        for sequence in batch:
            for step in sequence:
                current_state, action, reward, next_state, done = step

                # Convert each experience to tensors
                current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0)
                reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
                done_tensor = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

                # Compute target Q-value using the target network
                with torch.no_grad():
                    # Use the primary Q-network to select the best action for the next state
                    next_action = torch.argmax(self.q_network(next_state_tensor)).item()
                    # Use the target Q-network to calculate the Q-value for the selected action
                    future_q_value = self.target_q_network(next_state_tensor)[0, next_action].item()
                    target_q_value = reward + self.discount_factor * future_q_value * (1 - done)

                # Compute current Q-value
                q_values = self.q_network(current_state_tensor)
                current_q_value = q_values[0, action]

                # Compute loss and update the primary Q-network
                loss = self.loss_fn(current_q_value, torch.tensor(target_q_value, dtype=torch.float32))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def __call__(self, dt, stats):
        """
        Produce the dictionary of fixed scalar signals for
        each of the Asset instances within the Universe.
        """
        assets = self.universe.get_assets(dt)
        strategy_curve = stats['equity_curve']

        if self.weights is None:
            self.weights = {asset: 0.0 for asset in assets}
            self.start_date = dt

        end_date = (dt - pd.Timedelta(days=1)).date()
        start_date = (self.start_date - pd.Timedelta(days=1)).date()
        benchmark_curve = self.benchmark_curve[start_date:end_date].copy()
        # Performance Output
        tearsheet = TearsheetStatistics(
            strategy_equity=strategy_curve,
            benchmark_equity=benchmark_curve,
            title='HMM WMA vs SPY ETF'
        )

        # Dump results
        stats_benchmark = tearsheet.get_primary_results(tearsheet.get_results(benchmark_curve), 'benchmark')
        stats_strategy = tearsheet.get_primary_results(tearsheet.get_results(strategy_curve), 'QStrategy')

        cum_returns = stats_strategy['TotalReturns']
        profit_factor = stats_strategy['ProfitFactor']
        sharpe_strategy = stats_strategy['Sharpe']

        # Get the current state
        current_state = self._get_state(dt)

        # Choose an action
        action = self._choose_action(current_state)

        # Interpret the action
        action_name = self.action_mapping[action]
        reward = stats_strategy['TotalReturns'] - stats_benchmark['TotalReturns'] - 0.02

        # Update the counter for consecutive "long" actions
        if action_name == "long":
            self.consecutive_long_actions += 1
        else:
            self.consecutive_long_actions = 0  # Reset the counter if the action is not "long"

        # Apply a penalty if the threshold for consecutive "long" actions is exceeded
        #if self.consecutive_long_actions > self.max_consecutive_long and self.eval_mode == False:
        #    penalty = -1  # Example penalty formula
        #    reward += penalty
        #    print(f"Penalty applied for {self.consecutive_long_actions} consecutive 'long' actions: {penalty:.2f}")

        print(f"Epoch {self.epoch_number}, Date {dt}, Training {not self.eval_mode}, Executing action: {action_name}, Returns: {cum_returns:.2f}, Sharpe: {stats_strategy['Sharpe']:.2f}, Profit Factor: {profit_factor:.2f}, Benchmark: {stats_benchmark['TotalReturns']:.2f}, Reward: {reward:.2f}")

        # Assign weights based on the action
        if action_name == "long" and not self.isInvested:
            self.weights[assets[0]] = 1.0
            self.weights[assets[1]] = 0.0
            self.isInvested = True
        elif action_name == "short" and self.isInvested:
            self.weights[assets[0]] = 0.0
            self.weights[assets[1]] = 1.0
            self.isInvested = False

        # Update Q-network only if not in evaluation mode
        self._update_q_network(reward, current_state, self.previous_action)

        # Update previous state and action
        self.previous_state = current_state
        self.previous_action = action

        # Example: Update the target network every 10 episodes
        if self.counter % 5 == 0:
            self.update_target_network()

        #self.update_target_network_soft()

        self.counter += 1

        # Update the performance_stats DataFrame
        self.performance_stats.loc[(dt, self.epoch_number, not self.eval_mode), :] = {
            'Action': action_name,
            'Returns': round(cum_returns, 2),
            'Sharpe': round(stats_strategy['Sharpe'], 2),
            'Profit Factor': round(profit_factor, 2),
            'Benchmark Returns': round(stats_benchmark['TotalReturns'], 2),
            'Reward': round(reward, 2),
            'Training': not self.eval_mode
        }

        return self.weights

    def save_model(self, file_path):
        """
        Save the Q-network model to a file.

        Parameters
        ----------
        file_path : str
            The path to save the model.
        """
        try:
            torch.save(self.q_network.state_dict(), file_path)
            print(f"Model saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, file_path):
        """
        Load the Q-network model from a file.

        Parameters
        ----------
        file_path : str
            The path to load the model from.
        """
        if os.path.exists(file_path):
            try:
                self.q_network.load_state_dict(torch.load(file_path))
                self.q_network.eval()  # Set the model to evaluation mode
                print(f"Model loaded successfully from {file_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found at {file_path}")