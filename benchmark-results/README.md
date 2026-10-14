# Vector store benchmark results

Raw results of `VectorStoreBenchmark` (see AGENTS.md, "Vector store benchmark"), one directory per
benchmark session: `<date>-<machine>`. The data set is `wikipedia-embedding-gemma-2` (popular
English Wikipedia pages, EmbeddingGemma 2, 768 dimensions, 100 reference queries).

Each directory has three CSV files per store and corpus size (`<store>-<corpus>-*.csv`):

| file | rows | columns |
|---|---|---|
| `-run.csv` | 1 | `timestamp`, `device`, `store`, `corpus`, `dimension`, `insert_ns` (total), `get_100_ids_ns`, `embeddings_exact` (the store returned all 768 floats bit-exact), `reopen_and_get_all_ids_ns` (persistent stores), `disk_bytes` (stores in files) |
| `-searches.csv` | one per search | `search` (`k10`, `k100`, `k1000`: top-K; `k10_filtered`: top-10 among the records with metadata `bucket` = `0`, every 10th row), `query` (query id), `latency_ns`, `returned`, `hits_at_10` (of the first 10 results, how many are true top-10 neighbors), `ids` (all returned corpus row ids, space-separated, best first) |
| `-inserts.csv` | one per batch of 1,000 records | `first_row`, `rows`, `latency_ns` |

`summary.csv` is derived: `summarize.py` recomputes latency percentiles, recall@10 and insert
rates from the raw files and checks that they are complete:

```bash
python3 benchmark-results/summarize.py benchmark-results/*/
```

`plot.py` draws two charts per directory from the same data, as SVG (committed) and high-resolution
PNG (git-ignored): `latency` (median top-10 search latency by corpus size) and `recall`
(recall@10 per search type at the largest corpus size). It needs matplotlib; `uv run` installs it
on the fly:

```bash
uv run benchmark-results/plot.py benchmark-results/*/
```

Recall details: a result counts as a hit if it is a true top-10 neighbor, or if its similarity is
within 1e-5 of the 10th neighbor's (a tie that float rounding decides). The true neighbors come
from the data set's ground truth for that corpus size (the first N rows); for the filtered
search, the benchmark computes them exactly on the device.

## Adding a session

1. Run the benchmarks (commands in AGENTS.md). On a device, the Android Gradle plugin copies the
   CSV files to `Android/src/app/build/outputs/connected_android_test_additional_output/`;
   JVM runs write to `Android/src/app/build/outputs/vector-store-benchmark/`.
2. Copy them into a new `<date>-<machine>` directory, run `summarize.py` on it, and note the
   conditions below.

## Sessions

- `2026-10-14-emulator-pixel10-api37`: Android emulator (Pixel 10 AVD, Android 17, x86_64, 4 cores)
  on a Linux workstation; good for comparing stores, not for absolute numbers. Stores:
  `SqliteVectorStore` (MediaPipe, the app's current store), `ObjectBoxVectorStore` (ObjectBox
  5.4.2, default HNSW parameters) and `ObjectBoxVectorStoreWideSearch` (same, searching at least
  100 candidates).
- `2026-10-14-jvm-linux-workstation`: the same benchmark on the JVM of the workstation (ObjectBox
  with its Linux library, plus `InMemoryVectorStore`, an exact search that validates the recall
  computation); SQLite cannot run there.

Both sessions ran the code of the commit that added them.
