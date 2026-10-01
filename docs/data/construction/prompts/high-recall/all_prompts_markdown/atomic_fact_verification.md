# Atomic Fact Verification

## Meta-Information

- Source prompt family: [`prompts/eval.py`](../prompts/eval.py)
- Prompt family name: `ATOMIC_FACT_VERIFICATION_PROMPTS`
- Runtime generator: [`evaluation/atomic_fact_judge.py`](../evaluation/atomic_fact_judge.py)
- Active version: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{evidence_blueprint}`
- `{positive_documents}`

### Output tags parsed from the model response

- `<VERIFICATION>`
- `<SUMMARY>`
- nested tags such as:
  - `<EXPECTED_FACT>`
  - `<VERDICT>`
  - `<OBSERVATION>`
  - `<TOTAL_FACTS>`
  - `<FOUND>`
  - `<PARTIAL>`
  - `<NOT_FOUND>`
  - `<COVERAGE>`

### Parsed fields written back to the pipeline state

- `atomic_fact_verification`
- `atomic_fact_total`
- `atomic_fact_found`
- `atomic_fact_partial`
- `atomic_fact_not_found`
- `atomic_fact_coverage`
- `atomic_fact_doc_verdicts`

## Actual Prompt Template Sent To The LLM

```text
### I. Role

You are a Precision Fact Verification System. Your task is to verify that each atomic fact from the Evidence Blueprint appears in its corresponding generated document.

### II. Task

For each atomic fact specified in the Evidence Blueprint (inside `<DOC_N>` tags), verify whether that exact fact or its semantic equivalent appears in the corresponding `<DOCUMENT_N>` in the Generated Documents.

### III. Verification Guidelines

1. **Exact Match Preferred**: Look for the exact string or a very close paraphrase.
2. **Semantic Equivalence**: If the exact string is not present but the same information is conveyed, mark as FOUND.
3. **Partial Match**: If only part of the required fact is present, mark as PARTIAL.
4. **Not Found**: If the fact is missing or contradicted, mark as NOT_FOUND.

### IV. Output Structure

You must output verification for each document:

<VERIFICATION>
    <DOC_1>
        <EXPECTED_FACT>[The atomic fact from the blueprint]</EXPECTED_FACT>
        <VERDICT>FOUND, PARTIAL, or NOT_FOUND</VERDICT>
        <OBSERVATION>[Brief explanation of where/how the fact was found or why it's missing]</OBSERVATION>
    </DOC_1>
    <DOC_2>
        <EXPECTED_FACT>[The atomic fact from the blueprint]</EXPECTED_FACT>
        <VERDICT>FOUND, PARTIAL, or NOT_FOUND</VERDICT>
        <OBSERVATION>[Brief explanation]</OBSERVATION>
    </DOC_2>
    <DOC_N>
        <EXPECTED_FACT>[The atomic fact from the blueprint]</EXPECTED_FACT>
        <VERDICT>FOUND, PARTIAL, or NOT_FOUND</VERDICT>
        <OBSERVATION>[Brief explanation]</OBSERVATION>
    </DOC_N>
</VERIFICATION>

<SUMMARY>
    <TOTAL_FACTS>[Number of facts checked]</TOTAL_FACTS>
    <FOUND>[Count of FOUND verdicts]</FOUND>
    <PARTIAL>[Count of PARTIAL verdicts]</PARTIAL>
    <NOT_FOUND>[Count of NOT_FOUND verdicts]</NOT_FOUND>
    <COVERAGE>[Percentage of FOUND facts]</COVERAGE>
</SUMMARY>

### V. Input Data

**Evidence Blueprint:**
{evidence_blueprint}

**Generated Documents:**
{positive_documents}
```

