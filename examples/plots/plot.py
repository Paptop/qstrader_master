from matplotlib import cm, pyplot as plt
from matplotlib.dates import MonthLocator, YearLocator
import numpy as np


def plot_in_sample_hidden_states(hmm_model, df, rets):
    """
    Plot the adjusted closing prices masked by 
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    hidden_states = hmm_model.predict(rets)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        hmm_model.n_components, 
        sharex=True, sharey=True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df.index[mask], 
            df["EQ:SPY"][mask], 
            ".", linestyle='none', 
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()

def plot_sp500_with_regime(df, target):
    """
    Plots the S&P 500 index with color-coded regimes.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' prices and a 'regime' column.
    """
    plt.figure(figsize=(12, 6))

    # Define colors for regimes
    #colors = {0: 'red', 1: 'yellow', 2: 'blue', 3: 'pink', 4: 'black'}
    colors = {'Bull': 'red', 'Neutral': 'yellow', 'Bear': 'blue'}

    # Plot each regime separately
    for regime, data in df.groupby('pregime_label'):
        #plt.plot(data.index, data['EQ:SPY'], color=colors.get(regime, 'blue'), label=f'Regime {regime}', linewidth=2)
        plt.scatter(data.index, data[target], color=colors.get(regime, 'black'), 
                    label=f'Regime {regime}', s=10, alpha=0.5)

    plt.xlabel('Date')
    plt.ylabel('S&P 500 Index')
    plt.title('S&P 500 Index with Regime-Based Coloring')
    plt.legend()
    plt.show()