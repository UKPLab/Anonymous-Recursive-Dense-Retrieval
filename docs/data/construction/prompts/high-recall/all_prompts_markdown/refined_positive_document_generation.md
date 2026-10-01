# Refined Positive Document Generation

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `REFINED_POSITIVE_DOCUMENT_GENERATION_PROMPTS`
- Runtime generator: [`data_gen/refined_pos_gen.py`](../data_gen/refined_pos_gen.py)
- Active version: `default`
- Note: this is the legacy multi-document refinement prompt

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{num_documents}`
- `{word_count}`
- `{feedback}`
- `{few_shot_examples}`
- `{high_recall_query}`
- `{evidence_blueprint}`

### Output tags parsed from the model response

- `<DOCUMENTS>`
- `<AUDIT>`

### Parsed fields written back to the pipeline state

- `positive_documents`
- `initial_positive_documents`
- `retrieved_style_docs`

## Actual Prompt Template Sent To The LLM

```text
### I. Role

You are a knowledge base curator tasked with precisely generating {num_documents} **Independent Positive Documents** to support the provided High Recall Query. Your *only* source of truth for content is the detailed **Evidence Blueprint** provided below.

### II. Disentanglement & Stylistic Constraints (The Goal)

The document set MUST strictly adhere to the following:
1. **Atomic Fact Embedding:** Each document MUST contain the **EXACT STRING** of the fact/data point assigned to it in the Evidence Blueprint. This string must be present character-for-character to allow for automated verification.
2. **Data Disentanglement (MuSiQue Style):** The facts across the {num_documents} documents must be entirely compartmentalized and non-overlapping. No single document may contain information, dates, or figures sufficient to execute the **specific task** alone.
3. **Implicit Retrieval:** Frame the required fact within a dense, natural language context. Avoid explicitly stating "This is the answer for Document X."
4. **Mimetic Style Transfer:** Use the examples provided in Section V as templates for tone, vocabulary, and structural formatting (e.g., news snippets, technical memos). However, DO NOT use any factual data from the examples. Only use data/facts from the Evidence Blueprint.
5. **Independence & Length:** Each document must be a unique block of non-overlapping text (at least {word_count} words) that can stand alone.

### Feedback from Previous Attempts (If Any)
{feedback}

### III. Planning & Reasoning (The Blueprint)

**Output Requirement:** You must use the strict XML structure defined below. First, plan the content of each document to ensure it meets the disentanglement constraints and stylistic alignment.

    <REASONING>
        <BLUEPRINT_ANALYSIS>
            [Analyze the Evidence Blueprint and confirm assignment of facts to documents]
        </BLUEPRINT_ANALYSIS>
        <STYLE_ALIGNMENT_PLAN>
            [Identify which style from Section VI you will apply to which document]
        </STYLE_ALIGNMENT_PLAN>
        <DISENTANGLEMENT_PLAN>
            [Explain how you will ensure no single document reveals the full answer]
        </DISENTANGLEMENT_PLAN>
    </REASONING>

### IV. Mandatory Audit & Self-Correction

**Output Requirement:** You must audit the generated documents against the constraints. You must **always** output the full <AUDIT> block. 

Use the following logic for the "STATUS" tags:
- If a check passes, output: **COMPLIANT**
- If a check fails, output: **NON-COMPLIANT**

<AUDIT>
    <CHECK_PRECISION>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Did each document contain the EXACT STRING of its assigned fact?]</OBSERVATION>
    </CHECK_PRECISION>
    <CHECK_DISENTANGLEMENT>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Is it impossible to answer the query with just one document?]</OBSERVATION>
    </CHECK_DISENTANGLEMENT>
    <CHECK_DOCUMENT_COUNT>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Did you generate exactly {num_documents} documents?]</OBSERVATION>
    </CHECK_DOCUMENT_COUNT>
    <CHECK_STYLE_ADHERENCE>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Do the documents mirror the structure and tone of the few-shot examples without leaking their facts?]</OBSERVATION>
    </CHECK_STYLE_ADHERENCE>
</AUDIT>

### V. Stylistic Archetypes (Few-Shot Examples)
Use the following examples only to guide the structure, tone, and formatting of your output:
{few_shot_examples}

### V. Final Output Generation

After completing your reasoning and mandatory audit, provide the final documents.

**IMPORTANT:** You must include ALL three sections: the reasoning section (wrapped in <REASONING> and </REASONING>), the audit section (wrapped in <AUDIT> and </AUDIT>), AND the final documents (wrapped in <DOCUMENTS> and </DOCUMENTS> tags).
**NEGATIVE CONSTRAINT:** Do NOT repeat the input instructions or any conversational text. Start your response immediately with the <REASONING> tag.

<REASONING>
    [Content from Section III]
</REASONING>
<AUDIT>
    [Content from Section IV]
</AUDIT>
<DOCUMENTS>
    <DOCUMENT_1>
    [Full text of Document 1]
    </DOCUMENT_1>
    ...
    <DOCUMENT_{num_documents}>
    [Full text of Document {num_documents}]
    </DOCUMENT_{num_documents}>
</DOCUMENTS>

### Input Context:

**High Recall Query:** {high_recall_query}
**Evidence Blueprint (Derived from Query Reasoning):**
{evidence_blueprint}    
```

