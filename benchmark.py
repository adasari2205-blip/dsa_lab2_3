"""Generate six reproducible datasets and benchmark all three index structures.

Warning: the 500,000-word BST case can take significantly longer because this is
an ordinary (unbalanced) BST. The script uses shuffled insert order to avoid
deliberately sorted insertion.
"""
import csv
import gc
import random
import string
import time
from pathlib import Path

from index_structures import DictIndex, TrieIndex, BSTIndex, build_index, write_index

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
RESULTS_FILE = ROOT / "benchmark_results.csv"
VOCABULARY_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
MIN_TOTAL_WORDS = 1_000_100
WORDS_PER_LINE = 20
SEED = 20261012


def make_word(number):
    # Unique printable-ASCII tokens; no separator characters are used.
    return "w" + format(number, "06x")


def generate_dataset(distinct_count, filename, seed):
    rng = random.Random(seed)
    vocabulary = [make_word(i) for i in range(distinct_count)]
    # Each vocabulary item occurs at least once; fill to >1 million occurrences.
    total = max(MIN_TOTAL_WORDS, distinct_count)
    ids = list(range(distinct_count))
    rng.shuffle(ids)
    occurrences = [vocabulary[i] for i in ids]
    occurrences.extend(vocabulary[rng.randrange(distinct_count)]
                       for _ in range(total - distinct_count))
    rng.shuffle(occurrences)

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="ascii", newline="\n") as output:
        for start in range(0, len(occurrences), WORDS_PER_LINE):
            output.write(" ".join(occurrences[start:start + WORDS_PER_LINE]) + "\n")
    return vocabulary


def time_search(index, queries):
    start = time.perf_counter()
    found = 0
    for word in queries:
        if index.search(word) is not None:
            found += 1
    return time.perf_counter() - start, found


def run():
    DATA_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    results = []

    for distinct_count in VOCABULARY_SIZES:
        input_file = DATA_DIR / f"words_{distinct_count}.txt"
        print(f"\nDataset: {distinct_count:,} distinct words")
        vocabulary = generate_dataset(distinct_count, input_file, SEED + distinct_count)
        queries = [vocabulary[rng.randrange(distinct_count)] for _ in range(100_000)]

        for name, index_class in (
            ("Trie", TrieIndex),
            ("BST", BSTIndex),
            ("Dictionary", DictIndex),
        ):
            print(f"  Building {name}...")
            index = index_class()
            start = time.perf_counter()
            build_index(index, str(input_file))
            construction_seconds = time.perf_counter() - start

            search_seconds, found = time_search(index, queries)

            output_file = DATA_DIR / f"temp_{name.lower()}_{distinct_count}.txt"
            start = time.perf_counter()
            write_index(index, str(output_file))
            printing_seconds = time.perf_counter() - start
            output_file.unlink(missing_ok=True)

            results.append({
                "distinct_words": distinct_count,
                "implementation": name,
                "construction_seconds": round(construction_seconds, 6),
                "100000_searches_seconds": round(search_seconds, 6),
                "printing_seconds": round(printing_seconds, 6),
                "queries_found": found,
            })
            print(
                f"    build={construction_seconds:.2f}s, "
                f"search={search_seconds:.2f}s, print={printing_seconds:.2f}s"
            )

            del index
            gc.collect()

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved benchmark results to: {RESULTS_FILE}")


if __name__ == "__main__":
    run()
