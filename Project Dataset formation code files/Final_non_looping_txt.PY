import pandas as pd

# File paths
file1 = r"D:\Major Project from 27th March 2025\Combined_non_looping_regions_chr2_all pairs.txt" # Replace with actual file path
file2 = r"D:\Major Project from 27th March 2025\Combined_non_looping_regions_chr2_gap pairs.txt"# Replace with actual file path
output_txt = "combined_non_looping_chromosome2_data.txt" 

# Load TXT files (tab-separated)
df1 = pd.read_csv(file1, sep="\t", header=None, dtype=str)  # Read as string to avoid data type issues
df2 = pd.read_csv(file2, sep="\t", header=None, dtype=str) 

# Combine both TXT files 
combined_df = pd.concat([df1, df2], ignore_index=True)   

# Define columns: Assuming each file has 7 columns 
column_names = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"]
combined_df.columns = column_names 

# Remove duplicate rows based on all columns
cleaned_df = combined_df.drop_duplicates(subset=["chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"])

# Save the cleaned data to a TXT file (tab-separated)
cleaned_df.to_csv(output_txt, sep="\t", index=False, header=False)

print(f"Combined and cleaned data saved to {output_txt}")
