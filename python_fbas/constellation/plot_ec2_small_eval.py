# This just generates the plot of figure 3a

import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = "python_fbas/constellation/results-small.csv"
df = pd.read_csv(file_path)

plt.rcParams.update({
    'font.size': 16,         # Base font size
    'axes.titlesize': 22,    # Title font size
    'axes.labelsize': 22,    # X and Y label font size
    'xtick.labelsize': 20,   # X-axis tick labels font size
    'ytick.labelsize': 20,   # Y-axis tick labels font size
    'legend.fontsize': 20    # Legend font size
})

# Define custom legend labels
custom_legend = {
    "neighbors_yifan.json": "Constellation",
    "neighbors_random_6.json" : "Random degree 6",
    "neighbors_random_2.json" : "Random degree 2",
    "neighbors_random_4.json" : "Random degree 4",
    "neighbors_random_16.json" : "Random degree 16",
    "neighbors_all_to_all.json": "Fully connected"
}

# Plot data with increased line thickness
plt.figure(figsize=(10, 6))

for json_file, group in df.groupby("JSON File"):
    group = group.sort_values("Network Delay")  # Ensure data is sorted before plotting
    label = custom_legend.get(json_file, json_file)  # Use custom legend if available
    plt.plot(group["Network Delay"], group["Max TX Rate"], marker='o', linestyle='-', linewidth=2.5, label=label)

# Formatting the plot
plt.xlabel("Network Delay")
plt.ylabel("Max TX Rate")
# plt.title("Max TX Rate vs. Network Delay (Thicker Lines)")
plt.legend(title="")
plt.grid(True)

# Show the plot
plt.show()
