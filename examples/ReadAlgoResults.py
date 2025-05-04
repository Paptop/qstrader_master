import pandas as pd

# Specify the path to your CSV file
csv_file_path = 'performance_stats_2025-05-04-14:10:22.893.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
import matplotlib.pyplot as plt

df = df[df['Epoch']==155]
# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plot strategy returns vs benchmark returns with monthly detailed X axis
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Returns'], label='Strategy Returns', color='blue')
plt.plot(df['Date'], df['Benchmark Returns'], label='Benchmark Returns', color='orange')

# Add short and long actions as dots
short_actions = df[df['Action'] == 'short']
long_actions = df[df['Action'] == 'long']

plt.scatter(short_actions['Date'], short_actions['Returns'], color='red', label='Short Actions', marker='o')
plt.scatter(long_actions['Date'], long_actions['Returns'], color='green', label='Long Actions', marker='o')

# Add labels for each action dot with corresponding month, positioned slightly higher
for _, row in short_actions.iterrows():
    plt.text(row['Date'], row['Returns'] + 0.05, row['Date'].strftime('%m'), color='red', fontsize=8, ha='center', va='bottom')

for _, row in long_actions.iterrows():
    plt.text(row['Date'], row['Returns'] + 0.05, row['Date'].strftime('%m'), color='green', fontsize=8, ha='center', va='bottom')

plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Strategy Returns vs Benchmark Returns with Actions')
plt.legend()
plt.grid()

# Format the X axis to show monthly ticks
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

plt.xticks(rotation=45)
plt.show()