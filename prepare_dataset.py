import os
from datasets import Dataset, DatasetDict, Audio, concatenate_datasets
import glob

# --- Configuration ---
# Directory containing the source folders
# Use the directory where this script is located
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_FINALES_DIR = os.path.join(SOURCE_DIR, 'datos_finales')
OUTPUT_DIR = os.path.join(SOURCE_DIR, 'mms_finetuning_data')

# Constants
SAMPLING_RATE = 16000
TEST_SIZE_PERCENT = 0.1  # 10% for testing synthetic data
SEED = 42  # For reproducibility

def load_audio_paths_from_path(full_folder_path, category_label):
    """
    Loads all .wav files from a specific full path.
    Returns a list of dictionaries with 'audio' (path) and 'category' (label).
    """
    if not os.path.exists(full_folder_path):
        print(f"Warning: Folder not found at {full_folder_path}. Returning empty list.")
        return []

    audio_files = glob.glob(os.path.join(full_folder_path, "*.wav"))
    print(f"Found {len(audio_files)} files in '{os.path.basename(full_folder_path)}'")
    
    return [{"audio": f, "category": category_label} for f in audio_files]

def main():
    print("--- Starting Data Pipeline for Meta MMS Fine-Tuning ---")
    
    # 1. Load 'trainClean_wav' (Synthethic TTS) -> was 'original'
    print("\n1. Loading 'trainClean_wav' (Original/Synthetic)...")
    original_path = os.path.join(DATOS_FINALES_DIR, 'trainClean_wav')
    original_data = load_audio_paths_from_path(original_path, 'original')
    
    if not original_data:
        print(f"Error: No data found in {original_path}. Cannot proceed.")
        return

    ds_original = Dataset.from_list(original_data)
    
    # 2. Split 'original' into 90% Train / 10% Test
    print(f"2. Splitting 'original' into {100 - TEST_SIZE_PERCENT*100}% Train and {TEST_SIZE_PERCENT*100}% Test...")
    split_dataset = ds_original.train_test_split(test_size=TEST_SIZE_PERCENT, seed=SEED)
    
    train_synthetic = split_dataset['train']
    test_synthetic = split_dataset['test']  # This is Metric A
    
    print(f"   -> Train Synthetic: {len(train_synthetic)} samples")
    print(f"   -> Test Synthetic (Metric A): {len(test_synthetic)} samples")

    # 3. Load 'trainAugmentation_wav' (Augmented Data) -> was 'augment'
    print("\n3. Loading 'trainAugmentation_wav' (Augmented)...")
    augment_path = os.path.join(DATOS_FINALES_DIR, 'trainAugmentation_wav')
    augment_data = load_audio_paths_from_path(augment_path, 'augment')
    
    if augment_data:
        ds_augment = Dataset.from_list(augment_data)
        # 4. Mix 'augment' into Training Set
        train_combined = concatenate_datasets([train_synthetic, ds_augment])
        print(f"   -> Added {len(ds_augment)} augmented samples to training.")
    else:
        train_combined = train_synthetic
        print("   -> No augmented data found. Using only synthetic original for training.")

    print(f"   -> Final Training Set: {len(train_combined)} samples")

    # 5. Load 'reales' (Real Native Speaker Data)
    # Checking both root 'reales' and 'datos_finales/reales' just in case
    print("\n4. Loading 'reales' audios...")
    reales_path_root = os.path.join(SOURCE_DIR, 'reales')
    
    # We check root first, then datos_finales if needed.
    # But user said they don't have them yet. Let's stick to root 'reales' as placeholder 
    # or allow flexibility.
    real_data = load_audio_paths_from_path(reales_path_root, 'reales')
    
    if real_data:
        test_real = Dataset.from_list(real_data) # This is Metric B
        print(f"   -> Real Test Set (Metric B): {len(test_real)} samples")
    else:
        # Create an empty dataset with the same features
        test_real = Dataset.from_dict({"audio": [], "category": []})
        print("   -> No real data found (Metric B will be empty).")

    # 6. Assemble DatasetDict
    splits = {
        'train': train_combined,
        'test_synthetic': test_synthetic, # Metric A
    }

    if len(test_real) > 0:
        splits['test_real'] = test_real            # Metric B
    else:
        print("   -> Skipping 'test_real' in DatasetDict because it is empty (to avoid save errors).")

    final_dataset = DatasetDict(splits)

    # 7. Cast Audio Feature
    # This automatically loads and resamples the audio when accessed
    print(f"\n5. Casting audio column to {SAMPLING_RATE}Hz...")
    final_dataset = final_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # 8. Save to Disk
    print(f"\n6. Saving final DatasetDict to '{OUTPUT_DIR}'...")
    final_dataset.save_to_disk(OUTPUT_DIR)
    
    print("\n--- Pipeline Completed Successfully ---")
    print("Dataset Structure:")
    print(final_dataset)
    
if __name__ == "__main__":
    main()
