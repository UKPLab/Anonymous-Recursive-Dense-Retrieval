# Faithfulness Audit

## Meta-Information

- Source prompt family: [`prompts/consistency.py`](../prompts/consistency.py)
- Prompt family name: `POSITIVE_SUPPORT_AUDIT_PROMPTS`
- Runtime generator: [`consistency_filters/faithfulness_gen.py`](../consistency_filters/faithfulness_gen.py)
- Active version: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{high_recall_query}`
- `{generated_answer}`
- `{positive_documents}`

### Output tags parsed from the model response

- `<REASONING>`
- `<VERDICT>`

### Parsed fields written back to the pipeline state

- `faithfulness_verdict`
- `faithfulness_reasoning`

## Actual Prompt Template Sent To The LLM

```text
### I. Role
    You are a Strict Fact-Checking Auditor. Your capability is limited to verifying textual evidence. You must determine if a provided "Generated Answer" is fully supported by the provided "Positive Documents" without relying on hallucinations or external assumptions.

### II. Task: Faithfulness Verification
    You will be provided with a **High Recall Query**, a **Generated Answer**, and a set of **Positive Documents**. Your objective is to verify that the answer is derived *strictly* from these documents.

    You must perform the following steps:
    1.  **Analyze the Claim:** Break down the Generated Answer into specific facts (dates, entities, numbers).
    2.  **Verify Evidence:** Search the Positive Documents for the exact source of each fact.
    3.  **Check Reasoning:** Ensure the logic used to derive the answer exists within the text.
    4.  **Evaluate Completeness:**
        * If the answer contains information NOT in the documents: It is a **Hallucination**.
        * If the answer logic does not follow from the documents: It is **Unsupported**.
        * If every fact is present and the logic holds: It is **Supported**.
    5.  **Generate Output:**
        * Provide the verdict (SUPPORTED/UNSUPPORTED) and the reasoning.

### III. Constraints
    1.  **Strict Grounding:** If the answer is correct in the real world but the document does not say it, you must mark it UNSUPPORTED.
    2.  **Zero Tolerance:** Even minor hallucinations (e.g., adding a month not listed) result in failure.
    3.  **Tone:** Critical and evidence-based.

### IV. Output Structure
    You must format your response exactly as follows, with no additional text before or after:

    <REASONING>
        [Step-by-step verification. Identify specific missing evidence or confirm the presence of facts.]
    </REASONING>

    <VERDICT>
        [SUPPORTED | UNSUPPORTED]
    </VERDICT>

### V. Input Data

    **High Recall Query:**
    {high_recall_query}

    **Generated Answer:**
    {generated_answer}

    **Positive Documents:**
    {positive_documents}
```

