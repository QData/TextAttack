import os
import json
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Path to directory containing the result .jsonl files
results_dir = "./results"  # Change if needed

# Collect data from all .jsonl files
data = []

for filename in os.listdir(results_dir):
    if filename.startswith("toxic_") and filename.endswith(".jsonl"):
        path = os.path.join(results_dir, filename)
        
        # Parse filename for metadata
        parts = filename.split("_")
        if len(parts) < 5: continue  # sanity check
        
        transformation = parts[1]
        try:
            perturb_budget = int(parts[-1].replace("perturbs", "").replace(".jsonl", ""))
        except ValueError:
            continue

        # Load and analyse file content
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        total = len(lines)
        correct = sum(1 for row in lines if row.get("perturbed_output", 0.0) >= 0.5)
        percent_correct = (correct / total * 100) if total > 0 else 0

        data.append({
            "transformation": transformation,
            "perturb_budget": perturb_budget,
            "percent_correct": percent_correct
        })

# Create a DataFrame
df = pd.DataFrame(data)

# Pivot data for plotting
pivot_df = df.pivot(index="perturb_budget", columns="transformation", values="percent_correct")

# Plot
plt.figure(figsize=(10, 6))
for col in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[col], marker="o", label=col)

plt.title("Classification Accuracy vs Perturbation Budget")
plt.xlabel("Perturbation Budget")
plt.ylabel("Percentage Classified Correctly (%)")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 100)
plt.grid(True)
plt.legend(title="Transformation")
plt.tight_layout()
plt.show()

# save to png
output_path = os.path.join(results_dir, "accuracy_vs_perturbation.png")
plt.savefig(output_path)
plt.show()

print(f"Saved plot to: {output_path}")