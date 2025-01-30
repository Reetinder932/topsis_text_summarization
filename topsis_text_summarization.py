import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "Model": ["BART", "T5", "Pegasus", "GPT-3", "XLNet"],
    "ROUGE-1": [0.75, 0.78, 0.73, 0.76, 0.72],   
    "BLEU": [0.68, 0.72, 0.67, 0.71, 0.65],       
    "Inference Time (s)": [1.2, 1.5, 1.1, 1.3, 1.8],  
    "Model Size (MB)": [1200, 850, 1300, 1400, 1000]  
}

df = pd.DataFrame(data)
print("Initial Data:\n", df)

scaler = MinMaxScaler()

df_normalized = df.copy()
df_normalized["ROUGE-1"] = scaler.fit_transform(df[["ROUGE-1"]])
df_normalized["BLEU"] = scaler.fit_transform(df[["BLEU"]])
df_normalized["Inference Time (s)"] = scaler.fit_transform(-df[["Inference Time (s)"]])  
df_normalized["Model Size (MB)"] = scaler.fit_transform(-df[["Model Size (MB)"]])  

print("\nNormalized Data:\n", df_normalized)

weights = np.array([0.4, 0.3, 0.2, 0.1])  # ROUGE-1, BLEU, Inference Time, Model Size


weighted_matrix = df_normalized.iloc[:, 1:] * weights
print("\nWeighted Normalized Matrix:\n", weighted_matrix)

# Step 5: Calculate Ideal and Negative-Ideal Solutions
ideal_solution = weighted_matrix.max().values  # Best values
negative_ideal_solution = weighted_matrix.min().values  # Worst values

# Step 6: Calculate Euclidean Distances
dist_to_ideal = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
dist_to_negative_ideal = np.sqrt(((weighted_matrix - negative_ideal_solution) ** 2).sum(axis=1))

# Step 7: Calculate Relative Closeness to Ideal Solution
relative_closeness = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

# Add rankings to the DataFrame
df["TOPSIS Score"] = relative_closeness
df["Rank"] = df["TOPSIS Score"].rank(ascending=False)

print("\nFinal Rankings:\n", df)

# Save Results to CSV
df.to_csv("topsis_text_summarization_results.csv", index=False)

# Step 8: Visualize Results
import matplotlib.pyplot as plt
plt.barh(df["Model"], df["TOPSIS Score"], color="skyblue")
plt.xlabel("TOPSIS Score")
plt.ylabel("Model")
plt.title("Model Rankings using TOPSIS")
plt.gca().invert_yaxis()

# Save the chart as a PNG file
plt.savefig("model_rankings.png", dpi=300, bbox_inches="tight")

# Display the chart
plt.show()