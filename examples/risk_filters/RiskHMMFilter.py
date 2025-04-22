import random
import numpy as np
import pandas as pd
from examples.risk_filters.HMM import train_hmm
from qstrader.risk_model.risk_model import RiskModel

def generate_decreasing_weights(prices):
        """
        Returns a NumPy array of weights that decrease from 1.0 to a positive value
        in equally spaced steps, matching the length of `prices`.
        
        Example for 4 prices:
            n = 4
            step = 1/4 = 0.25
            weights = [1.0, 0.75, 0.5, 0.25]
        """
        n = len(prices)
        step = 1 / n #0.25
        # For i in [0, 1, 2, ..., n-1], each weight = 1.0 - (i * step).
        weights = 1.0 - np.arange(n) * step
        weights = np.clip(weights, 0, None)
        return weights

class RiskHMMFilter(RiskModel):
    def __init__(
        self,
        hmm_model,
        universe,
        adj_price_returns,
        alpha_model,
        params,
        isFuture = False,
        isInvested = False
    ):
        self.hmm_model = hmm_model
        self.universe = universe
        self.adj_price_returns = adj_price_returns
        self.alpha_model = alpha_model
        self.prev_regime = 1
        self.retrain_month = 1
        self.number_of_bulls = 0
        self.isInvested = isInvested

        #parameters for current day switch
        self.isFuture = isFuture
        self.smooth = params['smooth']
        self.easing = params['easing']
        self.lag = params['lag']
        self.current_day_only = params['current_day_only']
        self.random = params['random']
        self.smooth_days = params['smooth_days'] # default 19
        self.lag_in_days = params['lag_in_days'] # default 0
        self.window_size = params['window_size'] # default 365
        self.params = params


    def train_model(self, start_hmm_dt, end_hmm_dt):
        return train_hmm(start_hmm_dt, end_hmm_dt, self.adj_price_returns)
        
    def predict_future_regime(self, returns, steps_ahead=1):
        """
        Predicts the most probable future regime after 'steps_ahead' days.
        
        :param returns: Array of past returns.
        :param steps_ahead: How many steps into the future to predict.
        :return: Most probable future regime.
        """
        # Get current hidden state
        hidden_state = self.hmm_model.predict(returns)[-1]

        # Get transition matrix
        trans_mat = self.hmm_model.transmat_

        # Compute future state probability distribution
        future_prob = np.linalg.matrix_power(trans_mat, steps_ahead)[hidden_state]

        # Predict the most probable future state
        future_state = np.argmax(future_prob)

        return future_state

    def _determine_current_regime(self, dt):
        """
        Determines the predicted regime by making a prediction
        on the adjusted closing returns from the price handler
        object and then taking the final entry integer as
        the "hidden regime state".
        """

        # Past
        start_date = (dt - pd.Timedelta(days=self.window_size)).date()
        # Future
        end_date = (dt + pd.Timedelta(days=0)).date()

        prices = np.array(self.adj_price_returns[start_date:end_date]['EQ:SPY'])
        if prices.size == 0:
            return None

        returns = np.column_stack(
            [prices]
        )

        if self.lag:
            predicted_hidden_states = self.hmm_model.predict(returns)
            hidden_states = predicted_hidden_states[:-self.lag_in_days].mean()
            hidden_state = 0 if hidden_states_mean < 0.5 else 1
            #return hidden_state

        if self.smooth:
            predicted_hidden_states = self.hmm_model.predict(returns)
            hidden_states_mean = predicted_hidden_states[-self.smooth_days:][::-1].mean()
            hidden_state = 0 if hidden_states_mean < 0.5 else 1
            #return hidden_state
        
        if self.easing:
            predicted_hidden_states = self.hmm_model.predict(returns)
            hidden_states = predicted_hidden_states[-self.smooth_days:][::-1]
            ww = generate_decreasing_weights(hidden_states)
            hh = np.average(hidden_states, weights=ww)
            hidden_state = 0 if hh < 0.5 else 1
            #return hidden_state

        if self.random:
            predicted_hidden_states = [self._get_random_state() for _ in range(len(prices))]
            hidden_state = predicted_hidden_states[-1]

        if self.current_day_only:
            predicted_hidden_states = self.hmm_model.predict(returns)
            hidden_state = predicted_hidden_states[-1]

        if dt.date() in self.adj_price_returns.index:
            self.adj_price_returns.loc[start_date:end_date, 'regime'] = predicted_hidden_states
        else:
            print(f"Date {dt.date()} not found in DataFrame index.")
        return hidden_state
    
    def _determine_future_regime(self, dt, peek_steps=1, days_to_include=365):
        """
        Determines the predicted regime by making a prediction
        on the adjusted closing returns from the price handler
        object and then taking the final entry integer as
        the "hidden regime state".
        """

        # Past
        start_date = (dt - pd.Timedelta(days=days_to_include)).date()
        # Future
        end_date = (dt + pd.Timedelta(days=0)).date()

        prices = np.array(self.adj_price_returns[start_date:end_date]['EQ:SPY'])
        if prices.size == 0:
            return None

        returns = np.column_stack(
            [prices]
        )

        hidden_state = self.predict_future_regime(returns, steps_ahead=peek_steps)

        if dt.date() in self.adj_price_returns.index:
            self.adj_price_returns.loc[dt.date(), 'regime'] = hidden_state
        else:
            print(f"Date {dt.date()} not found in DataFrame index.")
        return hidden_state
    
    def _get_random_state(self):
        return random.choice([0, 1])
    
    def _adjust_weights_on_current_switch_regime(self, dt, weights):
        ######################
        # CURRENT DAY SWITCH #
        ######################
        start_hmm_dt = dt - pd.Timedelta(days=365*20)
        assets = self.universe.get_assets(dt)

        if self.retrain_month % 365 == 0:
            self.hmm_model = self.train_model(start_hmm_dt, dt)

        #if self.retrain_month % 21 > 0:
        #    self.retrain_month += 1
        #    return weights
        #self._determine_current_regime(dt, easing=True, smooth_days=3)
        #self._determine_current_regime(dt, smooth=True, smooth_days=10, window_size=365)
        #regime= self._determine_future_regime(dt, peek_steps=1, days_to_include=1000)
        regime = self._determine_current_regime(dt)
        #print(regime)

        if self.prev_regime is None:
            self.prev_regime = regime
            return weights
        
        # 0 is bull
        # 1 is bear
        # 0 < 1 < 2 < 3 < 4
        # 1 > 0
        #bullSwitch = (regime == 0 or regime == 1 or regime == 2) and (self.prev_regime == 3 or self.prev_regime == 4)
        #bearSwitch = (regime == 4 or regime == 3 or regime == 2) and (self.prev_regime == 0 or self.prev_regime == 1)#regime > self.prev_regime
        bullSwitch = regime < self.prev_regime
        bearSwitch = regime > self.prev_regime

        #self.train_model(start_hmm_dt, dt)

        #if self.alpha_model.isSelfInvested is False:


        # If in the desirable regime, let buy and sell orders
        # work as normal for a long-only trend following strategy
        if bullSwitch and not self.isInvested: #and:
            weights[assets[0]] = 1.0
            weights[assets[1]] = 0.0
            print(dt, "Risk model: Adjusting to regime 0")
            self.isInvested = True
            #self.train_model(start_hmm_dt, dt)

        # If in the undesirable regime, do not allow any buy orders
        # and only let sold/close orders through if the strategy
        # is already invested (from a previous desirable regime)
        if bearSwitch and self.isInvested: #and market_weight >= bonds_weight:
            weights[assets[0]] = 0.0
            weights[assets[1]] = 1.0
            self.isInvested = False
            print(dt, "Risk model: Adjusting to regime 1")
            #self.train_model(start_hmm_dt, dt)
        
        self.prev_regime = regime
        self.retrain_month += 1
        return weights
    
    def _adjust_weights_on_future_switch_regime(self, dt, weights):

        #################
        # FUTURE SWITCH #
        #################
        assets = self.universe.get_assets(dt)
        start_hmm_dt = dt - pd.Timedelta(days=365 * 5)
        curr_regime = self._determine_current_regime(dt)
        fut_regime = self._determine_future_regime(dt, peek_steps=1)

        if self.prev_regime is None:
            self.prev_regime = curr_regime
            return weights
        
        print("Current: ", curr_regime, "Future: ", fut_regime)
        bullSwitch = fut_regime < curr_regime
        bearSwitch = fut_regime > curr_regime
        # 0 --> 1 Bear
        # 1 --> 0 Bull
        #bullSwitch = self.prev_regime > 0.4 * future_regime + 0.6 * curr_regime
        #bearSwitch = self.prev_regime < 0.4 * future_regime + 0.6 * curr_regime
        # If in the desirable regime, let buy and sell orders
        # work as normal for a long-only trend following strategy
        if bullSwitch and not self.isInvested: #and:
            weights[assets[0]] = 1.0
            weights[assets[1]] = 0.0
            print(dt, "Adjusting to regime 0")
            #self.alpha_model.isSelfInvested = True
            self.isInvested = True
            #self.train_model(start_hmm_dt, dt)

        # If in the undesirable regime, do not allow any buy orders
        # and only let sold/close orders through if the strategy
        # is already invested (from a previous desirable regime)
        if bearSwitch and self.isInvested: #and market_weight >= bonds_weight:
            weights[assets[0]] = 0.0
            weights[assets[1]] = 1.0
            #self.alpha_model.isSelfInvested = False
            self.isInvested = False
            print(dt, "Adjusting to regime 1")
            #self.train_model(start_hmm_dt, dt)
        
        self.prev_regime = curr_regime
        return weights
    
    def __call__(self, dt, weights):
        if self.isFuture:
            return self._adjust_weights_on_future_switch_regime(dt, weights)
        return self._adjust_weights_on_current_switch_regime(dt, weights)
    
    def get_description(self):
        return {
            "name": "RiskRegimeFilter",
            "description": "RiskRegimeFilter",
            "future": self.isFuture,
            "parameters": self.params
        }