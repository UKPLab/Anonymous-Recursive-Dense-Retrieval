# Prompt Inventory

This folder documents the active prompt templates in the original `high-recall-pipeline`.

## How To Read These Files

Each file is split into three separate sections:

1. `Meta-Information`
   This is repository-level information. It is **not** part of the prompt sent to the LLM.

2. `Runtime Injection Contract`
   This lists the placeholder variables that get injected into the prompt at runtime, plus the tags/fields parsed out of the model output.

3. `Actual Prompt Template Sent To The LLM`
   This is the literal active template text from the codebase, with placeholders such as `{high_recall_query}` left intact.

## Prompt Files

- [persona_generation.md](./persona_generation.md)
- [query_generation.md](./query_generation.md)
- [positive_document_generation.md](./positive_document_generation.md)
- [single_positive_document_generation.md](./single_positive_document_generation.md)
- [single_typed_positive_document_generation.md](./single_typed_positive_document_generation.md)
- [refined_positive_document_generation.md](./refined_positive_document_generation.md)
- [single_refined_positive_document_generation.md](./single_refined_positive_document_generation.md)
- [negative_document_generation.md](./negative_document_generation.md)
- [solver.md](./solver.md)
- [faithfulness_audit.md](./faithfulness_audit.md)
- [leakage_audit.md](./leakage_audit.md)
- [memorization_blind.md](./memorization_blind.md)
- [memorization_judge.md](./memorization_judge.md)
- [eval_judge.md](./eval_judge.md)
- [atomic_fact_verification.md](./atomic_fact_verification.md)

## Important Clarification

For negative generation, the actual runtime input is sample-level:

- the full query
- the full evidence blueprint
- one assigned positive fact
- the other blueprint facts to avoid
- one source positive document to mimic
- one sampled failure-mode instruction

So the current negative-generation design is now per-positive-document and the outputs are reassembled into a `negative_documents` block.
