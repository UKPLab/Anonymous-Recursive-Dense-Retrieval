# Reproducing the paper

Each configuration in [`configs/`](configs/) covers a group of tables. The runner performs **one traversal per
search** and reads it out with every readout that the tables compare. This follows the paper's protocol
(Appendix F): readouts are compared on identical states. The aggregator compares the result with the paper's
values, which are stored in the configuration.

| Config | Tables | Suites | Metric |
|---|---|---|---|
| `multihop_tree.yaml` | 1, 3b, 15 (prefixes), 19, 20, 21 | MuSiQue, BrowseComp+, FanOutQA, FRAMES | Recall@5 (@10) |
| `multihop_inference_forms.yaml` | 3a, 15 (breadth), 30, 32, 33 | same | Recall@5 |
| `singlehop_tree.yaml` | 2, 5, 6 | BRIGHT, NanoBEIR, ToolRet, CoIR, TopiOCQA | nDCG@10 |
| `singlehop_readouts.yaml` | 3, 24, 27 | same, CoIR on the 17,901-query sample | nDCG@10 |
| `singlehop_continuation.yaml` | 14 | BRIGHT, NanoBEIR, ToolRet, TopiOCQA | nDCG@10 |
| `singlehop_breadth.yaml` | 2 (ablation) | BRIGHT, NanoBEIR, ToolRet, CoIR, TopiOCQA | nDCG@10 |

The package does not reproduce everything in the paper:

- Training (the RRW data and recipes are released separately).
- LLM reranking (Appendix J).
- Baselines with other encoders (ReasonIR, BGE-Reasoner, Octen, ChatRetriever, CoveR, GRITHopper's own method).
- BRIGHT-Pro, which uses the benchmark's official evaluation code.

## 1. Install

```bash
pip install -e ".[eval]"
```

## 2. Data

- **Single-hop suites** load from the Hugging Face Hub. They use the paper's repositories, splits, pinned
  revisions and instruction strings (`rdr.evaluation.benchmarks`):
  - `sentence-transformers/NanoBEIR-en`
  - `mteb/BRIGHT` (pinned revision, with its excluded ids)
  - the CoIR datasets
  - `mangopy/ToolRet-*`
- **TopiOCQA and the multi-hop suites** read prepared files from `--data-root`:

| Suite | Directory | Files |
|---|---|---|
| MuSiQue | `musique/` | `musique.json` (questions with `paragraphs` and `is_supporting`), `musique_corpus.json` (`title`, `paragraph_text`) |
| BrowseComp+ | `browsecomp/` | `doc_ids.json`, `doc_texts.json`, `queries.tsv` (`qid\tquery`), `qrel_evidence.txt` (TREC qrels, evidence documents) |
| FanOutQA | `fanoutqa/` | `fanoutqa_docs.jsonl`, `fanoutqa_dev_corpus.json`, `queries.tsv`, `qrel_source_groups.tsv` (`qid\tgroup\tdoc\trel`) |
| FRAMES | `frames/` | `frames.json` (questions with `groups`), `frames_corpus.json` (`{"docs": [{"id", "text"}]}`) |
| TopiOCQA | `topiocqa_1m/` | `queries.json` (`qid`, `conversation`, `gold`), `corpus.jsonl` (`id`, `title`, `text`; 1M passages) |

Gold labels follow Table 4 of the paper:

- MuSiQue credits supporting paragraphs.
- BrowseComp+ credits evidence documents.
- FanOutQA and FRAMES credit any passage of a relevant source group.

## 3. Run

```bash
python reproduce/run.py --config reproduce/configs/multihop_tree.yaml \
    --model RDR-8B --data-root data/ --out results/ \
    --encode-device cuda:0 --index-device cuda:1 [--shard 0 --num-shards 8] [--exact]
```

- **`--shard i --num-shards n`** splits the queries. Launch one process per GPU (pair).
- **Document embeddings** are cached per suite and task in `--index-cache`.
- **`--exact`** reproduces the paper's batch composition: fixed consecutive encoder batches, and states grouped
  per query (multi-hop) or per slot (single-hop). bf16 embeddings depend slightly on which texts share a batch.
  Without `--exact`, states are sorted by length, which is faster. The results then differ from the paper by
  small amounts.
- **Budgets.** Query states use up to 4096 tokens. Documents use 512 tokens (multi-hop passages) or 8192 tokens
  (single-hop corpora), as in the paper's runs; the configurations set both.

## 4. Compare with the paper

```bash
python reproduce/aggregate.py --config reproduce/configs/multihop_tree.yaml --results results/ --out table1.json
```

Each cell prints our value and its difference from the paper. A task is only aggregated when all of its shards
are present.

## Parity status

[`parity/`](parity/) holds the query-by-query parity checks of the package against the paper's evaluation code,
and `parity/PARITY_REPORT.md` summarizes them.
