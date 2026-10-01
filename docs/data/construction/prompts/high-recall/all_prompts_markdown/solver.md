# Solver

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `SOLVER_PROMPTS`
- Runtime generators:
  - [`data_gen/answer_gen.py`](../data_gen/answer_gen.py)
  - [`evaluation/baseline_gen.py`](../evaluation/baseline_gen.py)
- Active version: `2.0.0`
- Other versions present in source: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{feedback}`
- `{high_recall_query}`
- `{positive_documents}`

### Output tags parsed from the model response

- `<REASONING>`
- `<AUDIT>`
- `<ANSWER>`

### Parsed fields written back to the pipeline state

- `gold_answer`
- `reasoning`
- `audit`

Baseline variants also parse:

- `baseline_pos_neg_answer`
- `baseline_pos_only_answer`
- `baseline_neg_only_answer`

## Actual Prompt Template Sent To The LLM

```text
### I. Role

You are a Precision Question Answering System and Information Retrieval Auditor. Your capability is strictly limited to extracting and synthesizing information solely from the provided context.

### II. Task: Answer Generation & Verification (The Goal)

You will be provided with a **High Recall Query** and a set of **Positive Documents**. Your objective is to synthesize a comprehensive answer to the query based *only* on these documents.

**Constraints:**
1.  **Strict Grounding:** Do not use any internal knowledge.
2.  **Handling Insufficiency:** If the provided documents do not contain enough information to answer the query, your final answer must be exactly: `Insufficient information`.
3.  **Tone:** Analytical, objective, and concise.

### Feedback from Previous Attempts (If Any)
{feedback}

### III. Analysis & Reasoning

**Output Requirement:** You must use the strict XML structure defined below.

    <REASONING>
        <QUERY_ANALYSIS>
            [Identify the core question and constraints]
        </QUERY_ANALYSIS>
        <EVIDENCE_SYNTHESIS>
            [Scan documents and synthesize findings. Cite Document IDs.]
        </EVIDENCE_SYNTHESIS>
    </REASONING>

### IV. Mandatory Audit & Self-Correction

**Output Requirement:** You must audit the generated answer. You must **always** output the full <AUDIT> block.

Use the following logic for the "STATUS" tags:
- If a check passes, output: **COMPLIANT**
- If a check fails, output: **NON-COMPLIANT**

<AUDIT>
    <CHECK_GROUNDING>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Is every claim supported by the documents?]</OBSERVATION>
    </CHECK_GROUNDING>
    <CHECK_COMPLETENESS>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Did we answer the full query or correctly identify insufficiency?]</OBSERVATION>
    </CHECK_COMPLETENESS>
</AUDIT>

### V. Final Output Generation

After completing your reasoning and mandatory audit, provide the final answer.

**IMPORTANT:** You must include ALL three sections: the reasoning section (wrapped in <REASONING> and </REASONING>), the audit section (wrapped in <AUDIT> and </AUDIT>), AND the final answer (wrapped in <ANSWER> and </ANSWER> tags).
**NEGATIVE CONSTRAINT:** Do NOT repeat the input instructions or any conversational text. Start your response immediately with the <REASONING> tag.

<REASONING>
    [Content from Section III]
</REASONING>
<AUDIT>
    [Content from Section IV]
</AUDIT>
<ANSWER>
    [Your final answer text OR "Insufficient information"]
</ANSWER>

### Input Context:

**High Recall Query:**
{high_recall_query}

**Positive Documents:**
{positive_documents}
```

