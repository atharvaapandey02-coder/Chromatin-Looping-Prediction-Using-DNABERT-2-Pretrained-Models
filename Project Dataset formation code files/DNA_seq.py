import pandas as pd
import random
from Bio import SeqIO

# Load the reference genome
fasta_file = r"D:\Major Project\Human genome fasta files\chr2.fa"
genome_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

def get_anchor_sequence(chrom, start, end):
    """Extract 256 bp sequence centered at the mid-point."""
    if chrom not in genome_dict:
        return None
    
    mid_point = (start + end) // 2
    anchor_start = max(0, mid_point - 128)
    anchor_end = anchor_start + 256  # ensure exactly 256bp

    seq = genome_dict[chrom].seq[anchor_start:anchor_end]
    return str(seq) if len(seq) == 256 else None

def generate_combined_sequences(df, label_value):
    """Generate anchor1, anchor2, combined_sequence and label for the DataFrame."""
    df["anchor1"] = df.apply(lambda row: get_anchor_sequence(row["chrom1"], row["start1"], row["end1"]), axis=1)
    df["anchor2"] = df.apply(lambda row: get_anchor_sequence(row["chrom2"], row["start2"], row["end2"]), axis=1)

    # Only keep rows where both anchors are valid
    df = df.dropna(subset=["anchor1", "anchor2"])
    
    df["combined_sequence"] = df["anchor1"] + " [SEP] " + df["anchor2"]
    df["label"] = label_value
    return df[["combined_sequence", "label"]]

# Load looping and non-looping data
looping_df = pd.read_csv("Combined_looping_regions_chr2.txt", sep="\t", header=None,
                         names=["chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"])
non_looping_df = pd.read_csv("combined_non_looping_chromosome2_data.txt", sep="\t", header=None,
                             names=["chrom1", "start1", "end1", "chrom2", "start2", "end2", "score"])

# Ensure chromosome matches for both ends (very important)
looping_df = looping_df[looping_df["chrom1"] == looping_df["chrom2"]]
non_looping_df = non_looping_df[non_looping_df["chrom1"] == non_looping_df["chrom2"]]

# Generate sequences and labels
looping_processed = generate_combined_sequences(looping_df, label_value=1)
non_looping_processed = generate_combined_sequences(non_looping_df, label_value=0)

# Balance the dataset
min_size = min(len(looping_processed), len(non_looping_processed))
looping_balanced = looping_processed.sample(n=min_size, random_state=42)
non_looping_balanced = non_looping_processed.sample(n=min_size, random_state=42)

# Combine and shuffle
final_df = pd.concat([looping_balanced, non_looping_balanced]).reset_index(drop=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to file
output_file = "All_DNA_Sequences_chr2_balanced.txt"
with open(output_file, "w") as f:
    for seq, label in zip(final_df["combined_sequence"], final_df["label"]):
        f.write(f"{seq},{label}\n")

print(f" Final dataset saved to {output_file} with {len(final_df)} balanced entries.")














































