import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

# Read CSV and prepare
df = pd.read_csv("strategy_stats_over_time.csv")
print("CSV file successfully loaded.")
df['Strategy'] = df['Strategy'].str.replace(' vs SPY', '')
# Force all Benchmark rows to have risk_applied = False
df.loc[df['Strategy'].str.contains('Benchmark'), 'risk_applied'] = False

# Ensure datetime and period column
df['start_dt'] = pd.to_datetime(df['start_dt'])
df['end_dt'] = pd.to_datetime(df['end_dt'])
df['Period'] = df['start_dt'].dt.year.astype(str) + '–' + df['end_dt'].dt.year.astype(str)

# Remove duplicate Benchmark rows per period
df = df[~((df['Strategy'].str.contains('Benchmark')) &
          (df.duplicated(subset=['Period', 'Strategy'], keep='first')))]

# Define metrics to visualize
metrics = ['ProfitFactor', 'SharpeRatio', 'SortinoRatio', 'TotalReturns']
periods = sorted(df['Period'].unique())

# Create subplot grid: rows = periods, columns = metrics
n_rows = len(periods)
n_cols = len(metrics)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
axes = np.atleast_2d(axes)

# Plot
for i, period in enumerate(periods):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]

        # Filter and sort data
        df_period = df[df['Period'] == period].sort_values(by=metric, ascending=False)
        heat_data = df_period.set_index('Strategy')[[metric]]

        # Create heatmap annotations with optional star for risk-based models
        annotations = []
        for strategy in heat_data.index:
            value = heat_data.loc[strategy, metric]
            is_risk = df_period[df_period['Strategy'] == strategy]['risk_applied'].values[0]
            text = f"{value:.2f}" + (" ★" if is_risk else "")
            annotations.append([text])

        # Turn annotations into DataFrame (same shape as heat_data)
        annot_df = pd.DataFrame(annotations, index=heat_data.index, columns=[metric])

        # Plot heatmap
        sns.heatmap(
            heat_data,
            annot=annot_df,
            fmt="",
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar=False,
            linewidths=0.5,
            linecolor='gray'
        )

        # Force all annotation text to black (including stars)
        for text in ax.texts:
            text.set_color('white')
            text.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()
            ])
        
        # Make only the Benchmark label bold
        for tick_label in ax.get_yticklabels():
            label_text = tick_label.get_text()
            if "Benchmark" in label_text:
                tick_label.set_fontweight('bold')

        # Highlight Benchmark with red rectangle
        for k, strategy in enumerate(heat_data.index):
            if 'Benchmark' in strategy:
                rect = Rectangle((0, k), 1, 1, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

        ax.set_xlabel("")
        ax.set_ylabel("" if j > 0 else "Strategy")
        if i == 0:
            ax.set_title(metric)
        if j == 0:
            ax.set_ylabel(f"{period}", fontsize=12)

        # Make x-axis title (metric name) bold and larger — only on top row
        if i == 0:
            ax.set_title(metric, fontsize=12, fontweight='bold', color='black')

        # Make period label (y-axis title per row) bold — only on first column
        if j == 0:
            ax.set_ylabel(period, fontsize=12, fontweight='bold', color='black', labelpad=10)

    # Add horizontal lines between periods (rows)
for row in range(1, n_rows):  # start from row 1 to avoid a top line
    ypos = row / n_rows - 0.01
    line = plt.Line2D([0, 1], [1 - ypos, 1 - ypos], transform=fig.transFigure,
                      color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    fig.add_artist(line)

# Final layout
plt.tight_layout()
plt.savefig("strategy_metrics_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
