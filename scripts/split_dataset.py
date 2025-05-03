import os
import shutil
import random
import argparse

def split_data(source_dir, train_ratio=0.8):
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    random.shuffle(files)

    train_size = int(len(files) * train_ratio)
    train_files = files[:train_size]
    eval_files = files[train_size:]

    train_dir = os.path.join(source_dir, 'train')
    eval_dir = os.path.join(source_dir, 'eval')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    for f in train_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(train_dir, f))

    for f in eval_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(eval_dir, f))

    print(f"Moved {len(train_files)} files to '{train_dir}'")
    print(f"Moved {len(eval_files)} files to '{eval_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into training and evaluation sets.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the data to split")
    args = parser.parse_args()

    split_data(args.source_dir)
