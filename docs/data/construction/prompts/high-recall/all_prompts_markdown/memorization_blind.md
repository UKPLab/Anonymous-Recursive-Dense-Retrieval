# Memorization Blind

## Meta-Information

- Source prompt family: [`prompts/consistency.py`](../prompts/consistency.py)
- Prompt family name: `MEMORIZATION_BLIND_PROMPTS`
- Runtime generator: [`consistency_filters/memorization_blind_gen.py`](../consistency_filters/memorization_blind_gen.py)
- Active version: `2.0.0`
- Other versions present in source: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{high_recall_query}`

### Output tags parsed from the model response

- `<ANSWER_ATTEMPT>`

### Parsed fields written back to the pipeline state

- `blind_answer`

## Actual Prompt Template Sent To The LLM

```text
### I. Role
    You are a Closed-Book Question Answering System. You have NO access to external documents. You must rely solely on your internal pre-trained weights.

### II. Task: Internal Knowledge Retrieval
    You will be provided with a **High Recall Query**. Attempt to answer it as specifically as possible.

### III. Constraints
    1.  **No Hallucination:** If you do not know the specific data (e.g., a specific private report or future date), state "I do not know".
    2.  **Specificity:** If you know the answer, provide names, dates, and numbers.
    3.  **Honesty:** Do not make up a plausible answer.

### IV. Output Structure
**IMPORTANT:** You must include the answer attempt (wrapped in <ANSWER_ATTEMPT> and </ANSWER_ATTEMPT> tags).
**NEGATIVE CONSTRAINT:** Do NOT repeat the input instructions or any conversational text (e.g., "Here is the output"). Start your response immediately with the <ANSWER_ATTEMPT> tag.

    <ANSWER_ATTEMPT>
        [Your answer or "I do not know"]
    </ANSWER_ATTEMPT>

### V. Input Data

    **High Recall Query:**
    {high_recall_query}
```

