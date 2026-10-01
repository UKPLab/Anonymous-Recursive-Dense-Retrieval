# Negative Document Generation

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `NEGATIVE_DOCUMENT_GENERATION_PROMPTS`
- Runtime generator: [`data_gen/neg_gen.py`](../data_gen/neg_gen.py)
- Active version: `3.0.0`
- Other versions present in source: `default`, `2.0.0`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{word_count}`
- `{feedback}`
- `{high_recall_query}`
- `{evidence_blueprint}`
- `{assigned_fact}`
- `{other_blueprint_facts}`
- `{source_positive_document}`
- `{failure_mode}`

### Output tags parsed from the model response

- `<NEGATIVE_DOCUMENT>`
- `<AUDIT>`

### Parsed fields written back to the pipeline state

- `_single_negative_document`
- `_failure_mode_type`
- `_failure_mode_instruction`
- after assembly: `negative_documents`
- after assembly: `negative_failure_modes`

### Important structural fact

The active runtime is now per-positive-document:

- one positive document is selected
- one assigned fact is selected
- all other blueprint facts are provided as exclusions
- one source positive document is provided for style mimicry
- one failure mode is sampled
- one negative document is generated

The generator then reassembles these per-document negatives into a single `negative_documents` block for the sample.

## Actual Prompt Template Sent To The LLM

```text
### I. Role

You are an expert at generating "Hard Negative" documents for a High Recall Retrieval Benchmark. You are given one source positive document and must produce one corresponding negative document that remains topically relevant but is insufficient for solving the overall multi-hop problem.

### II. Task: Per-Document Hard Negative Generation

You must generate **exactly one hard negative document** aligned to the provided source positive document.

**Critical Constraints:**
1. **Style Mimicry:** Closely mimic the tone, structure, vocabulary, and surface form of the source positive document.
2. **Topic Relevance:** The output must remain highly related to the high recall query and look like a plausible search result.
3. **Assigned-Fact Exclusion:** The output must NOT contain, restate, paraphrase, or imply the assigned positive fact.
4. **Blueprint Safety:** The output must NOT contain, restate, paraphrase, or imply any other fact from the evidence blueprint. It must not create shortcuts for the multi-hop reasoning.
5. **Non-Contradiction:** Do NOT directly contradict the blueprint facts. Instead, present adjacent but ultimately non-answer-bearing information.
6. **Failure-Mode Compliance:** Apply the provided failure mode instruction while preserving plausibility.
7. **Independence & Length:** The document must stand alone and be at least {word_count} words long.

### Feedback from Previous Attempts (If Any)
{feedback}

### III. Planning & Reasoning

**Output Requirement:** You must use the strict XML structure defined below.

    <REASONING>
        <FAILURE_MODE_ANALYSIS>
            [Explain how the failure mode will be applied to this one source document.]
        </FAILURE_MODE_ANALYSIS>
        <SAFETY_PLAN>
            [Explain how you will avoid the assigned fact and all other blueprint facts without contradicting them.]
        </SAFETY_PLAN>
        <STYLE_PLAN>
            [Explain how you will mimic the source positive document's style.]
        </STYLE_PLAN>
    </REASONING>

### IV. Mandatory Audit & Self-Correction

**Output Requirement:** You must audit the generated document against the constraints. You must always output the full <AUDIT> block.

<AUDIT>
    <CHECK_RELEVANCE>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Does it still look query-relevant and plausible?]</OBSERVATION>
    </CHECK_RELEVANCE>
    <CHECK_ASSIGNED_FACT_EXCLUSION>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Did you avoid the assigned fact?]</OBSERVATION>
    </CHECK_ASSIGNED_FACT_EXCLUSION>
    <CHECK_BLUEPRINT_SAFETY>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Did you avoid all other blueprint facts and shortcuts?]</OBSERVATION>
    </CHECK_BLUEPRINT_SAFETY>
    <CHECK_NON_CONTRADICTION>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Did you avoid directly contradicting the blueprint?]</OBSERVATION>
    </CHECK_NON_CONTRADICTION>
</AUDIT>

### V. Final Output Generation

**IMPORTANT:** Include the reasoning, audit, and final document. Start immediately with the <REASONING> tag.

<REASONING>
    [Content from Section III]
</REASONING>
<AUDIT>
    [Content from Section IV]
</AUDIT>
<NEGATIVE_DOCUMENT>
    [Full text of the hard negative document]
</NEGATIVE_DOCUMENT>

### Input Context:

**High Recall Query:** {high_recall_query}
**Full Evidence Blueprint:** {evidence_blueprint}
**Assigned Positive Fact To Avoid:** {assigned_fact}
**Other Blueprint Facts To Avoid:** {other_blueprint_facts}
**Source Positive Document To Mimic:** {source_positive_document}
**Failure Mode Instruction:** {failure_mode}
```
