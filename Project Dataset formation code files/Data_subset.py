import os
import shutil
import random

# Modify these paths as per your directory structure
input_dir = "D:\Major Project from 27th March 2025\Chromosome_DNA seq_sets"
output_dir = "D:\Major Project from 27th March 2025\DNA Sequences subsets"

os.makedirs(output_dir, exist_ok=True)

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()

    if len(lines) < 20000:
        # If file has fewer than 20000 lines, copy as is
        shutil.copy(input_file_path, output_file_path)
        print(f"Copied (unchanged): {os.path.basename(input_file_path)}")
        return

    # Separate sequences by label
    label_0 = [line for line in lines if line.strip().endswith(",0")]
    label_1 = [line for line in lines if line.strip().endswith(",1")]

    # Ensure at least 10k of each label exists
    if len(label_0) < 10000 or len(label_1) < 10000:
        print(f"Skipping {os.path.basename(input_file_path)}: not enough sequences for both labels")
        shutil.copy(input_file_path, output_file_path)
        return

    # Randomly sample 10k from each label
    subset_0 = random.sample(label_0, 10000)
    subset_1 = random.sample(label_1, 10000)

    # Combine and shuffle
    subset = subset_0 + subset_1
    random.shuffle(subset)

    # Write to new output file
    with open(output_file_path, 'w') as outfile:
        outfile.writelines(subset)

    print(f"Subset created: {os.path.basename(output_file_path)}")

# Process all files in input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt") or filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_file(input_path, output_path)
