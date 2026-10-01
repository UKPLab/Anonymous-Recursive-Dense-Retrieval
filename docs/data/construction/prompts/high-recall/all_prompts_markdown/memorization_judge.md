# Memorization Judge

## Meta-Information

- Source prompt family: [`prompts/consistency.py`](../prompts/consistency.py)
- Prompt family name: `MEMORIZATION_JUDGE_PROMPTS`
- Runtime generator: [`consistency_filters/memorization_judge_gen.py`](../consistency_filters/memorization_judge_gen.py)
- Active version: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{high_recall_query}`
- `{blind_answer}`
- `{generated_answer}`

### Output tags parsed from the model response

- `<REASONING>`
- `<VERDICT>`

### Parsed fields written back to the pipeline state

- `memorization_verdict`
- `memorization_reasoning`

## Actual Prompt Template Sent To The LLM

```text
### I. Role
    You are a Contamination Judge. Your role is to determine if a query is "CONTAMINATED" (common knowledge) or "Clean" (requires retrieval).

### II. Task: Compare Answers
    You will be provided with a **Blind Answer** (generated from memory) and the **Gold Reference Answer** (generated from documents).

    You must perform the following steps:
    1.  **Compare Facts:** Check if the Blind Answer contains the *specific* entities, numbers, or lists found in the Gold Reference.
    2.  **Evaluate Memorization:**
        * If the Blind Answer is "I do not know" or factually wrong: The query is **CLEAN**.
        * If the Blind Answer matches the Gold Reference (even partially correct on specific data): The query is **CONTAMINATED**.
    3.  **Generate Output:** Verdict on contamination.

### III. Constraints
    1.  **Specifics Matter:** Vague knowledge (e.g., knowing a company exists) does NOT count as contamination. Knowing the *specific* financial report value DOES.
    2.  **Strictness:** If the model solves the query without docs, the query is invalid for our benchmark.

### IV. Output Structure
    You must format your response exactly as follows, with no additional text before or after:

    <REASONING>
        [Explain the overlap. Did the model know the specific secret data?]
    </REASONING>

    <VERDICT>
        [CLEAN | CONTAMINATED]
    </VERDICT>

### V. Input Data

    **High Recall Query:**
    {high_recall_query}

    **Blind Answer (Internal Knowledge):**
    {blind_answer}

    **Gold Reference Answer:**
    {generated_answer}
```

