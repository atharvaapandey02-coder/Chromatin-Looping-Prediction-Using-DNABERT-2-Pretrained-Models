import pandas as pd
import os

# User-defined parameters
interaction_file = r"D:\Major Project from 27th March 2025\HG00512_merged_replicates.e500.clusters.cis\HG00512_merged_replicates.e500.clusters.cis.BE3"  
output_directory = "Looping_regions_for_chromosomes2 (sub parts 60K bp)"  # Directory to save output files
chromosome_number = "2"  
min_difference = 58000  # Minimum base pair difference in a part 
max_difference = 64000  # Maximum base pair difference in a part 

# Ensure output directory exists 
os.makedirs(output_directory, exist_ok=True)

# Load the chromatin interaction file
df = pd.read_csv(interaction_file, sep="\t", header=None, names=[
    "chrom1", "start1", "end1", "chrom2", "start2", "end2", "score" 
])

# Filter interactions for the given chromosome
filtered_df = df[((df["chrom1"] == f"chr{chromosome_number}") | (df["chrom2"] == f"chr{chromosome_number}")) & (df["score"] >= 4) ]

# Sort interactions by the start coordinate of the left anchor sequence
filtered_df = filtered_df.sort_values(by=["start1"], ascending=True).reset_index(drop=True)

# Divide into sub-parts where the start1 difference is between 58K and 64K 
sub_parts = []
current_part = []
first_start = None

for index, row in filtered_df.iterrows():
    if first_start is None:
        first_start = row["start1"]

    current_difference = row["start1"] - first_start

    # If the max difference is exceeded, save the current part and start a new one
    if current_difference > max_difference:
        if current_difference >= min_difference:
         sub_parts.append(pd.DataFrame(current_part))  # Store current part
        current_part = []  # Start a new part 
        first_start = row["start1"]  

    current_part.append(row) 

# Ensure the last remaining interactions are also included as a sub-part
if current_part:
    sub_parts.append(pd.DataFrame(current_part)) 

# Save each part to separate TXT files 
for i, part in enumerate(sub_parts):
    output_file = os.path.join(output_directory, f"Looping_regions_chr{chromosome_number}_part{i+1}.txt")
    part.to_csv(output_file, sep="\t", index=False, header=False)
    # print(f"Saved: {output_file}")

print("All sub-parts processed and saved successfully!") 
