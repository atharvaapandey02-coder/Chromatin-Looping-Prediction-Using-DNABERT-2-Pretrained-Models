import pandas as pd
import networkx as nx
import os
import glob

# Directory containing looping region files
input_directory = r"D:\Major Project from 27th March 2025\Looping_regions_for_chromosomes2 (sub parts 60K bp)"
output_directory = r"D:\Major Project from 27th March 2025\Non_Looping_Regions(chromosome 2)_all pairs"  

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True) 

# Get all looping region files from the directory
looping_files = glob.glob(os.path.join(input_directory, "*.txt"))  

# Process each file separately
for interaction_file in looping_files: 
    filename = os.path.basename(interaction_file)  # Extract filename for naming output 

    # print(f"Processing: {filename}")

    # Read the looping interaction file
    df = pd.read_csv(interaction_file, sep="\t", header=None, names=[
        "chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"
    ], encoding="utf-8", engine="python")

    

    # Initialize an undirected graph for looping regions
    G = nx.Graph()

    # Add edges from chromatin interactions (looping regions)
    for _, row in df.iterrows():
        node1 = (row["chrom1"], row["start1"], row["end1"])
        node2 = (row["chrom2"], row["start2"], row["end2"])

        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2)

    # Create the complement graph (G') -> All possible non-looping regions
    G_complement = nx.complement(G)

    # Extract connected components from G' (non-looping regions)
    non_looping_regions = []
    for component in nx.connected_components(G_complement):
        sorted_regions = sorted(component, key=lambda x: x[1])  # Sort by start position
        for i in range(len(sorted_regions) - 1):
            chrom1, start1, end1, chrom2, start2, end2 = *sorted_regions[i], *sorted_regions[i + 1]
            
            # Ensure regions are on the same chromosome
            if chrom1 == chrom2:
                non_looping_regions.append((chrom1, start1, end1, chrom2, start2, end2, 0))

    # Find isolated nodes (completely non-interacting regions)
    isolated_nodes = [node for node in G.nodes if G.degree(node) == 0]
    for i in range(len(isolated_nodes) - 1):
        chrom1, start1, end1, chrom2, start2, end2 = *isolated_nodes[i], *isolated_nodes[i + 1]

        # Ensure regions are on the same chromosome
        if chrom1 == chrom2:
            non_looping_regions.append((chrom1, start1, end1, chrom2, start2, end2, 0))

    # Remove non-looping regions where one sequence is completely within the other, partially overlapping, or shorter than 256 bp
    def filter_contained_and_overlapping_regions(regions):
        filtered = []
        for region in regions:
            chrom1, start1, end1, chrom2, start2, end2, score = region

            # Check if left sequence is completely within the right sequence
            if (start1 >= start2 and end1 <= end2) or (start2 >= start1 and end2 <= end1):
                continue  # Skip this region

            # Check if left and right sequences are partially overlapping
            if (start1 <= start2 <= end1) or (start2 <= start1 <= end2):
                continue  # Skip this region

            # Check if either sequence is shorter than 256 bp
            if (end1 - start1 < 256) or (end2 - start2 < 256):
                continue  # Skip this region

            filtered.append(region)
        return filtered

    non_looping_regions = filter_contained_and_overlapping_regions(non_looping_regions)

    # Convert to DataFrame and save each file separately
    output_df = pd.DataFrame(non_looping_regions, columns=[
        "chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"
    ])

    output_file = os.path.join(output_directory, f"Non_Looping_{filename}")
    output_df.to_csv(output_file, sep="\t", index=False, header=False)  # Save without headers

    
    

print("Processing complete for all looping region files!") 
