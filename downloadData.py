import os
from datasets import load_dataset

dataset_dir = "./datasets"

wikitext_dir = os.path.join(dataset_dir, "wikitext")
print("Downloading WikiText-2...")
try:
    dataset = load_dataset(
        "wikitext", 
        "wikitext-2-raw-v1", 
        cache_dir=wikitext_dir 
    )
    print(f"WikiText-2 saved to: {wikitext_dir}")
    print(f"Train samples: {len(dataset['train'])}")
except Exception as e:
    print(f"Failed to download WikiText-2: {e}")