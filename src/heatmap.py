import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_battery_dataset.csv")

# Calculate correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Create heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")

# Add color bar
plt.colorbar()

# Set axis labels
plt.xticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns,
           rotation=90)

plt.yticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns)

plt.title("Correlation Heatmap of Battery Dataset")

plt.tight_layout()

# Save image
plt.savefig("heatmap.png", dpi=300)

# Display heatmap
plt.show()

print("Heatmap saved successfully as heatmap.png")