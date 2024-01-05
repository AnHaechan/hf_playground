# ```
# sudo pip3 install -r requirements.txt
# sudo python3 load_dataset.py
# ```

from datasets import load_dataset

# PILE = "EleutherAI/pile"
PILE = "monology/pile-uncopyrighted"
DISK_PATH = "/mnt/disks/persist/pile"

def print_examples(num_examples=3, split="validation", category="Github"):
    # Stream to iterate over each sample, without downloading the entire dataset
    pile_stream = load_dataset(PILE, streaming=True, split=split, cache_dir=DISK_PATH)

    # Only iterate e.g. { "pile_set_name": "Github" }
    pile_github_stream = pile_stream.filter(lambda sample: sample['meta']["pile_set_name"] == category)

    for _ in range(num_examples):
        print(next(iter(pile_github_stream)))

def download(split):
    # Load the whole dataset, as filtering before downloading is not allowed
    pile_split = load_dataset(PILE, streaming=False, split=split, cache_dir=DISK_PATH)
    
    # filter only Github + ArXiv
    pile_github_arxiv_split = pile_split.filter(lambda sample: sample['meta']["pile_set_name"] in ["Github", "ArXiv"])
    pile_github_arxiv_split.save_to_disk(f"{DISK_PATH}/pile_gtihub_arxiv_{split}.hf")

if __name__ == "__main__":
    print_examples()
    print_examples(2, "train", "ArXiv")

    for split in ["validation", "training", "test"]:
        download(split)
