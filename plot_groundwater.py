
import pandas as pd
import matplotlib.pyplot as plt

# Load the prepared monthly dataset
df = pd.read_csv("data/selected_station.csv")

df["Month"] = pd.to_datetime(df["Month"])

# Plot historical groundwater measurements
plt.figure(figsize=(14, 6))
plt.plot(
    df["Month"],
    df["ground_water_level"],
    linewidth=1
)

plt.title("Historical Groundwater Levels - Chitradurga Station 70202")
plt.xlabel("Year")
plt.ylabel("Groundwater Level (check source units)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the graph
plt.savefig("data/groundwater_history.png", dpi=300)
print("Graph saved to data/groundwater_history.png")

# Display the graph
plt.show()
