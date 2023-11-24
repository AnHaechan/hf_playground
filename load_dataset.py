from datasets import load_dataset, get_dataset_config_names, load_dataset_builder

# PILE = "EleutherAI/pile"
PILE = "monology/pile-uncopyrighted"

def print_examples(num_examples=3, split="validation", category="Github"):
    # Stream to iterate over each sample, without downloading the entire dataset
    pile_stream = load_dataset(PILE, streaming=True, split=split)

    # Only iterate e.g. { "pile_set_name": "Github" }
    pile_github_stream = pile_stream.filter(lambda sample: sample['meta']["pile_set_name"] == category)

    for _ in range(num_examples):
        print(next(iter(pile_github_stream)))

def download(split):

    # Load the whole dataset, as filtering before downloading is not allowed
    pile_split = load_dataset(PILE, streaming=False, split=split)
    
    for category in ["Github", "ArXiv"]:
        pile_category_split = pile_split.filter(lambda sample: sample['meta']["pile_set_name"] == category)
        pile_category_split.save_to_disk(f"pile_{category}_{split}.hf")

    # download merged
    pile_both_cat_split = pile_split.filter(lambda sample: sample['meta']["pile_set_name"] in ["Github", "ArXiv"])
    pile_both_cat_split.save_to_disk(f"pile_github_and_arxiv_{split}.hf")

if __name__ == "__main__":
    print_examples()
    print_examples(2, "train", "ArXiv")

    for split in ["training", "test", "validation"]:
        download(split)
