# ```
# sudo pip3 install -r requirements.txt
# sudo python3 load_dataset.py
# ```

from datasets import load_dataset, load_from_disk

# PILE = "EleutherAI/pile"
PILE = "monology/pile-uncopyrighted"
DISK_PATH = "/mnt/disks/persist2/pile"
NUM_PROC = 32

def print_examples(num_examples=3, split="validation", category="Github"):
    # Stream to iterate over each sample, without downloading the entire dataset
    pile_stream = load_dataset(PILE, streaming=True, split=split, cache_dir=DISK_PATH)

    # Only iterate e.g. { "pile_set_name": "Github" }
    pile_github_stream = pile_stream.filter(lambda sample: sample['meta']["pile_set_name"] == category)

    for _ in range(num_examples):
        print(next(iter(pile_github_stream)))

def download():
    # Load the whole dataset, as filtering before downloading is not allowed
    
    print("loading dataset")
    pile = load_dataset(PILE, streaming=False, cache_dir=DISK_PATH, num_proc=NUM_PROC)

    # filter only Github + ArXiv
    for split_name, pile_split in pile.items():
        # e.g. "train", pile_train
        print(f"filtering and saving {split_name}")
        pile_github_arxiv_split = pile_split.filter(lambda sample: sample['meta']["pile_set_name"] in ["Github", "ArXiv"], num_proc=NUM_PROC)
        pile_github_arxiv_split.save_to_disk(f"{DISK_PATH}/pile_github_arxiv_{split_name}.hf", num_proc=NUM_PROC)

def test_load_from_disk():
    print("loading from disk")
    pile_filtered_test = load_from_disk(f"{DISK_PATH}/pile_github_arxiv_train.hf")
    print(pile_filtered_test)
    #  print(pile_filtered_test[2399])
    # print(pile_filtered_test[3])

if __name__ == "__main__":
    # print_examples()
    # print_examples(2, "train", "ArXiv")

    # download()

    test_load_from_disk()