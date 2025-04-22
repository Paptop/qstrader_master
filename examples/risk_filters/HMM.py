
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.discriminant_analysis import StandardScaler

def train_hmm(start_date, end_date, data):
    # Create the Gaussian Hidden markov Model and fit it
    # to the SPY returns data, outputting a score
    train_data = data.loc[start_date.date():end_date.date()]
    rets = np.column_stack([train_data['EQ:SPY']])
    hmm_model = GaussianHMM(
       n_components=3, covariance_type="full", n_iter=500, tol=1e-4, random_state=42
    ).fit(rets)
    print("Model Score:", hmm_model.score(rets))
    return hmm_model

def train_hmm_on_data(train_data):
    # Create the Gaussian Hidden markov Model and fit it
    # to the SPY returns data, outputting a score
    hmm_model = GaussianHMM(
       n_components=3, covariance_type="full", n_iter=500, tol=1e-4, random_state=42
    ).fit(train_data)
    print("Model Score:", hmm_model.score(train_data))
    return hmm_model