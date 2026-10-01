# Leakage Audit

## Meta-Information

- Source prompt family: [`prompts/consistency.py`](../prompts/consistency.py)
- Prompt family name: `NEGATIVE_LEAKAGE_AUDIT_PROMPTS`
- Runtime generator: [`consistency_filters/leakage_gen.py`](../consistency_filters/leakage_gen.py)
- Active version: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{high_recall_query}`
- `{generated_answer}`
- `{negative_documents}`

### Output tags parsed from the model response

- `<REASONING>`
- `<VERDICT>`

### Parsed fields written back to the pipeline state

- `leakage_verdict`
- `leakage_reasoning`

### Important note

The runtime variable name is `{negative_documents}`, and the current pipeline passes the full combined negative-document block into it.

## Actual Prompt Template Sent To The LLM

```text
### I. Role
    You are a Negative Safety Auditor. Your goal is to identify "False Negatives" (documents that are supposed to be irrelevant but actually contain the correct answer). You must aggressively attempt to derive the "Target Answer" from the "Negative Documents".

### II. Task: Leakage Detection
    You will be provided with a **High Recall Query**, a **Target Answer** (the truth), and **Negative Documents**.

    You must perform the following steps:
    1.  **Analyze the Target:** Understand exactly what the correct answer is.
    2.  **Scan Negatives:** Search the Negative Documents to see if they contain this specific answer.
    3.  **Evaluate Leakage:**
        * If the documents allow you to derive the Target Answer: They are **DEFECTIVE** (Leakage).
        * If the documents contradict the Target Answer: They are **SAFE**.
        * If the documents are irrelevant or miss key constraints (e.g., wrong year): They are **SAFE**.
    4.  **Generate Output:** Determine if the documents leak the answer.

### III. Constraints
    1.  **Contextual Awareness:** A document is only "Leaking" if it answers the *specific* query constraints.
    2.  **Differentiation:** If the document gives a *different* answer than the Target (contradiction), it is SAFE. Leakage means finding the *same* answer.
    3.  **Tone:** Aggressive auditing.

### IV. Output Structure
    You must format your response exactly as follows, with no additional text before or after:

    <REASONING>
        [Explain if the target answer was found. If SAFE, explain why (e.g., "Discusses the topic but for the wrong year").]
    </REASONING>

    <VERDICT>
        [SAFE | LEAK]
    </VERDICT>

### V. Input Data

    **High Recall Query:**
    {high_recall_query}

    **Target Answer (Truth):**
    {generated_answer}

    **Negative Documents:**
    {negative_documents}
```
