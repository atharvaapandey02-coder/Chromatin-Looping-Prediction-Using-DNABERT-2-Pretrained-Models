import os
import glob

# Directory where all chromosome TXT files are stored
input_directory = r"D:\Major Project from 27th March 2025\Looping_regions_for_chromosomes2 (sub parts 60K bp)"
output_file = r"D:\Major Project from 27th March 2025\Combined_looping_regions_chr2.txt"  # Final combined file

# Find all TXT files in the directory 
txt_files = glob.glob(os.path.join(input_directory, "*.txt")) 

# Set to store unique lines (to remove duplicates)   
unique_lines = set()

# Read and merge all TXT files 
for file in txt_files:
    with open(file, "r") as infile:
        for line in infile: 
            unique_lines.add(line.strip())  # Add line to the set (removes duplicates automatically)

# Write the unique merged content to the output file
with open(output_file, "w") as outfile:
    outfile.write("\n".join(unique_lines) + "\n")  # Write unique lines back

print(f"All files merged with duplicates removed. Saved in: {output_file}")
