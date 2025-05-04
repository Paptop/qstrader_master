import random
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from examples.risk_filters.HMM import train_hmm_on_data
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

class MultiRiskHMMFilter(RiskModel):
    def __init__(
        self,
        universe,
        data,
        features,
        params,
        start_hmm_dt,
        end_hmm_dt
    ):
        self.hmm_model = None
        self.universe = universe
        self.prev_regime = None
        self.retrain_month = 1
        self.isInvested = False

        self.features = features
        self.start_hmm_dt = start_hmm_dt
        self.end_hmm_dt = end_hmm_dt

        # data and model
        self.data = data # data for all periods, for getting a price
        self.tData = self.transform_data_for_hmm(self.data) # transformed data for all periods for the model
        self.hmm_model = self.train_model(start_hmm_dt, end_hmm_dt)

        #parameters for current day switch
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
        # Prepare train data
        data = self.tData.loc[start_hmm_dt.date():end_hmm_dt.date()]
        return train_hmm_on_data(data)

    def transform_data_for_hmm(self, data):
        active_features = [feature for feature, par in self.features.items() if par['w'] > 0.0]
        rets = data[list(active_features)]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(rets)
        # Wrap scaled result back into a DataFrame with the same index and columns
        X_scaled_df = pd.DataFrame(X_scaled, index=rets.index, columns=rets.columns)
        return X_scaled_df

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

        data_to_predict = self.tData.loc[start_date:end_date].copy()

        if self.lag:
            predicted_hidden_states = self.hmm_model.predict(data_to_predict)
            hidden_states = predicted_hidden_states[:-self.lag_in_days].mean()
            hidden_state = 0 if hidden_states_mean < 0.5 else 1
            #return hidden_state

        if self.smooth:
            predicted_hidden_states = self.hmm_model.predict(data_to_predict)
            hidden_states_mean = predicted_hidden_states[-self.smooth_days:][::-1].mean()
            hidden_state = 0 if hidden_states_mean < 0.5 else 1
            #return hidden_state
        
        if self.easing:
            predicted_hidden_states = self.hmm_model.predict(data_to_predict)
            hidden_states = predicted_hidden_states[-self.smooth_days:][::-1]
            ww = generate_decreasing_weights(hidden_states)
            hh = np.average(hidden_states, weights=ww)
            hidden_state = 0 if hh < 0.5 else 1
            #return hidden_state

        if self.random:
            predicted_hidden_states = [self._get_random_state() for _ in range(len(data_to_predict))]
            hidden_state = predicted_hidden_states[-1]

        if self.current_day_only:
            predicted_hidden_states = self.hmm_model.predict(data_to_predict)
            hidden_state = predicted_hidden_states[-1]
        
        # Step 0: Separate features by type
        return_features = [f for f, meta in self.features.items() if meta['sort_by_returns']]
        volatility_features = [f for f, meta in self.features.items() if not meta['sort_by_returns']]

        # Step 1: Extract weights
        # Extract return weights with optional flipping
        return_weights = np.array([
            self.features[f]['w'] * (-1 if self.features[f].get('flip_values', False) else 1)
            for f in return_features
        ])

        # Extract volatility weights with optional flipping
        volatility_weights = np.array([
            self.features[f]['w'] * (-1 if self.features[f].get('flip_values', False) else 1)
            for f in volatility_features
        ])

        # Step 2: Assign regimes to data
        regime_series = pd.Series(predicted_hidden_states, index=data_to_predict.index)
        self.data.loc[data_to_predict.index, 'pregime'] = regime_series

        # Step 3: Compute mean and std per regime
        grouped_stats = self.data.groupby('pregime')[list(self.features.keys())].agg(['mean', 'std'])

        mean_df = grouped_stats.xs('mean', axis=1, level=1)
        std_df = grouped_stats.xs('std', axis=1, level=1)

        # Step 4: Compute weighted mean and std for scoring
        weighted_return_score = mean_df[return_features].values @ return_weights
        weighted_volatility_score = std_df[volatility_features].values @ volatility_weights

        # Step 5: Composite score = return strength - volatility penalty
        composite_score = weighted_return_score + weighted_volatility_score

        # Step 6: Create Series with regime index
        composite_score = pd.Series(composite_score, index=mean_df.index)
        sorted_states = composite_score.sort_values()

        # Step 4: Dynamically assign labels
        regime_labels = ['Bear'] + ['Neutral'] * (len(sorted_states) - 2) + ['Bull']

        # Step 5: Create mapping from state number to regime label
        state_regime_map = dict(zip(sorted_states.index, regime_labels))
        self.data.loc[data_to_predict.index, 'pregime_label'] = self.data.loc[data_to_predict.index, 'pregime'].map(state_regime_map)

        #if dt.date() in self.data.index.date:
        #    self.data.loc[start_date:end_date, 'regime'] = predicted_hidden_states
        #else:
        #    print(f"Date {dt.date()} not found in DataFrame index.")
        return hidden_state, state_regime_map
    
    def _adjust_weights_on_current_switch_regime(self, dt, weights):
        ######################
        # CURRENT DAY SWITCH #
        ######################
        start_hmm_dt = dt - pd.Timedelta(days=365*20)
        assets = self.universe.get_assets(dt)

        if self.retrain_month % 12 == 0:
            self.hmm_model = self.train_model(start_hmm_dt, dt)

        #self._determine_current_regime(dt, easing=True, smooth_days=3)
        #self._determine_current_regime(dt, smooth=True, smooth_days=10, window_size=365)
        #regime= self._determine_future_regime(dt, peek_steps=1, days_to_include=1000)
        regime, mappings = self._determine_current_regime(dt)

        regimeName = mappings[regime]
        #print(mappings)
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

        regimeChanged = regimeName != self.prev_regime
        bullSwitch = regimeChanged and regimeName == 'Bull'
        bearSwitch = regimeChanged and regimeName == 'Bear'

        #bullSwitch = (regimeName in ['Bull', 'Neutral']) and (self.prev_regime == 'Bear')
        #bearSwitch = (regimeName == 'Bear') and (self.prev_regime in ['Bull', 'Neutral'])

        #bullSwitch = regime < self.prev_regime
        #bearSwitch = regime > self.prev_regime

        # If in the desirable regime, let buy and sell orders
        # work as normal for a long-only trend following strategy
        if bullSwitch and not self.isInvested: #and:
            weights[assets[0]] = 1.0
            weights[assets[1]] = 0.0
            #print(dt, "Risk model: Adjusting to regime 0")
            self.isInvested = True
            #self.train_model(start_hmm_dt, dt)

        # If in the undesirable regime, do not allow any buy orders
        # and only let sold/close orders through if the strategy
        # is already invested (from a previous desirable regime)
        if bearSwitch and self.isInvested: #and market_weight >= bonds_weight:
            weights[assets[0]] = 0.0
            weights[assets[1]] = 1.0
            self.isInvested = False
            #print(dt, "Risk model: Adjusting to regime 1")
            #self.train_model(start_hmm_dt, dt)
        
        self.prev_regime = regime
        self.retrain_month += 1
        return weights
    
    def __call__(self, dt, weights):
        return self._adjust_weights_on_current_switch_regime(dt, weights)
    
    def get_description(self):
        return {
            "name": "Multimodal RiskRegimeFilter on array of feature",
            "description": "RiskRegimeFilter",
            "parameters": self.params,
            "features": self.features
        }