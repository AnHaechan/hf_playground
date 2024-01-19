from datasets import load_from_disk

DISK_PATH = "/mnt/disks/persist2/pile"
NUM_PROC = 32
MAX_LEN = 100_000

for split in ["test", "validation", "train"]:
    pile_filtered_test = load_from_disk(f"{DISK_PATH}/pile_github_arxiv_{split}.hf")

    for file_num, example in enumerate(pile_filtered_test):
        text = example['text']

        # TODO: batch the file IO, write only when reached MAX_LEN
        with open(f"{DISK_PATH}/raw_txt/{split}/{file_num}.txt", 'w') as file:
            file.write(text)