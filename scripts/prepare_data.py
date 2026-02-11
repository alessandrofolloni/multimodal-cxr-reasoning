"""
Prepare CheXpert dataset with synthetic clinical notes.
Generates notes for all samples and saves processed CSV files.
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Now we can import
exec(open(Path(__file__).parent.parent / "src/data/synthetic_notes.py").read())


def process_dataset(csv_path: Path, output_path: Path, dataset_name: str):
    """Process a dataset and add synthetic clinical notes.
    
    Args:
        csv_path: Path to input CSV (train.csv or valid.csv)
        output_path: Path to save processed CSV
        dataset_name: Name for logging (e.g., "train" or "valid")
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*80}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Fix paths (remove CheXpert-v1.0-small/ prefix)
    df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small/', '')
    
    # Pathology columns
    pathology_columns = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    # Initialize generator
    generator = SyntheticNoteGenerator()
    
    # Generate notes
    print("Generating synthetic clinical notes...")
    notes = []
    
    for idx in tqdm(range(len(df)), desc=f"Processing {dataset_name}"):
        row = df.iloc[idx]
        labels = {col: row[col] for col in pathology_columns if col in df.columns}
        note = generator.generate_note(labels)
        notes.append(note)
    
    # Add notes to dataframe
    df['clinical_note'] = notes
    
    # Save processed CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df)} samples to {output_path}")
    
    # Show statistics
    note_lengths = df['clinical_note'].str.split().str.len()
    print(f"\nNote statistics:")
    print(f"  Mean length: {note_lengths.mean():.1f} words")
    print(f"  Min length: {note_lengths.min()} words")
    print(f"  Max length: {note_lengths.max()} words")
    
    # Show examples
    print(f"\nExample notes from {dataset_name}:")
    for i in range(3):
        print(f"\n{i+1}. {df.iloc[i]['clinical_note']}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare CheXpert dataset with synthetic notes")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/chexpert",
        help="Directory containing CheXpert data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    print(f"CheXpert Data Preparation")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process train set
    train_csv = data_dir / "train.csv"
    train_output = output_dir / "train_with_notes.csv"
    train_df = process_dataset(train_csv, train_output, "train")
    
    # Process validation set
    valid_csv = data_dir / "valid.csv"
    valid_output = output_dir / "valid_with_notes.csv"
    valid_df = process_dataset(valid_csv, valid_output, "valid")
    
    print(f"\n{'='*80}")
    print("✅ Data preparation complete!")
    print(f"{'='*80}")
    print(f"Train: {len(train_df)} samples → {train_output}")
    print(f"Valid: {len(valid_df)} samples → {valid_output}")
    print(f"\nNext step: Use these CSV files for model training")


if __name__ == "__main__":
    main()