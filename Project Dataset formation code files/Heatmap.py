import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the CSV file
file_path = r"D:\Major Project from 27th March 2025\Meet 25th April\f1_score_matrix.csv"
df = pd.read_csv(file_path, index_col=0)

# 1. Determine the desired row‐order from the column names
ordered_rows = []
for col in df.columns:
    chrom = col.split('_')[0]
    num = chrom.replace('chr', '')
    row_idx = f"chr_{num}"
    if row_idx in df.index:
        ordered_rows.append(row_idx)

# 2. Reorder the DataFrame rows
df = df.loc[ordered_rows]

# 3. Sort the rows and columns numerically based on the chromosome number
def sort_chromosomes(chromosomes):
    # Extract numeric part of the chromosome names (e.g., 'chr1' becomes 1)
    return sorted(chromosomes, key=lambda x: int(''.join(filter(str.isdigit, x))))

# Sort the rows and columns
df = df.loc[sort_chromosomes(df.index)]
df = df[sort_chromosomes(df.columns)]

# 4. Plot the heatmap
# Make the heatmap square by setting width and height equal
num_rows, num_cols = df.shape
cell_size = 0.7  # Adjust this to make cells larger/smaller
fig_size = cell_size * max(num_rows, num_cols)
plt.figure(figsize=(fig_size, fig_size))

sns.set(font_scale=1.0)  # Slightly reduced overall scale to accommodate value text
ax = sns.heatmap(
    df,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 10}  # Font size for values inside the heatmap
)

plt.title("Heatmap of F1_score Metrics", fontsize=16)
plt.xlabel("Predicted on Chromosome (DNA seq)", fontsize=13)
plt.ylabel("Chromosome Model", fontsize=13)

# 5. Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 6. Save to the specified directory
output_path = r"D:\Major Project from 27th March 2025\Meet 25th April\F1_score_heatmap.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True) 
plt.tight_layout()
plt.savefig(output_path, dpi=300)  # dpi=300 for better clarity

# 7. Show it
plt.show()
