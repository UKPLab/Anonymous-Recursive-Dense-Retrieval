# Single Typed Positive Document Generation

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `SINGLE_POSITIVE_DOCUMENT_TYPED_PROMPTS`
- Runtime generator: [`data_gen/pos_gen.py`](../data_gen/pos_gen.py)
- Active version: `default`
- Used only when document-types mode is enabled

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{doc_type_name}`
- `{doc_type_description}`
- `{doc_type_example}`
- `{word_count}`
- `{feedback}`
- `{high_recall_query}`
- `{assigned_fact}`
- `{evidence_blueprint}`

### Output tags parsed from the model response

- `<DOCUMENT>`
- `<AUDIT>`

### Parsed fields written back to the pipeline state

- `_single_document`
- `audit`

## Actual Prompt Template Sent To The LLM

```text
### I. Role

You are a knowledge base curator tasked with generating **exactly 1 Independent Positive Document** to support the provided High Recall Query. Your *only* source of truth for factual content is the **Evidence Blueprint** provided below.

### II. Document Type Specification

You must format this document as: **{doc_type_name}**

**Formatting Instructions:**
{doc_type_description}

**Style Example (structure/tone only — do NOT copy these facts):**
{doc_type_example}

### III. Disentanglement Constraints

The document MUST strictly adhere to the following:
1. **Atomic Fact Embedding:** The document MUST contain the **EXACT STRING** of the fact/data point assigned to it (see "Your Assigned Fact" below). This string must be present character-for-character.
2. **Data Disentanglement:** This document must NOT contain information from other documents' blueprint entries. It must be impossible to answer the full query using only this document.
3. **Implicit Retrieval (Contextual Framing):** Frame the required fact/data point within a dense, natural language context (e.g., an excerpt from a technical brief or news article).
4. **Mimetic Style Transfer:** Follow the document type format specified above. Use the style example only for tone, structure, and formatting — do NOT reuse its facts.
5. **Independence & Length:** The document must be a unique block of text (at least {word_count} words) that can stand alone.

### Feedback from Previous Attempts (If Any)
{feedback}

### IV. Planning & Reasoning

**Output Requirement:** You must use the strict XML structure defined below.

    <REASONING>
        <BLUEPRINT_ANALYSIS>
            [Confirm your assigned fact and what info must be excluded]
        </BLUEPRINT_ANALYSIS>
        <STYLE_ALIGNMENT_PLAN>
            [Explain how you will apply the "{doc_type_name}" format to embed the fact naturally]
        </STYLE_ALIGNMENT_PLAN>
        <DISENTANGLEMENT_PLAN>
            [Explain how you will ensure the document stays disentangled from other docs]
        </DISENTANGLEMENT_PLAN>
    </REASONING>

### V. Mandatory Audit & Self-Correction

**Output Requirement:** You must audit the generated document. You must **always** output the full <AUDIT> block.

Use the following logic for the "STATUS" tags:
- If a check passes, output: **COMPLIANT**
- If a check fails, output: **NON-COMPLIANT**

<AUDIT>
    <CHECK_PRECISION>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Did the document contain the EXACT STRING of its assigned fact?]</OBSERVATION>
    </CHECK_PRECISION>
    <CHECK_DISENTANGLEMENT>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Is it impossible to answer the query with just this document?]</OBSERVATION>
    </CHECK_DISENTANGLEMENT>
    <CHECK_STYLE>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Does the document conform to the "{doc_type_name}" format?]</OBSERVATION>
    </CHECK_STYLE>
</AUDIT>

### VI. Final Output Generation

After completing your reasoning and mandatory audit, provide the final document.

**IMPORTANT:** You must include ALL three sections: the reasoning section (wrapped in <REASONING> and </REASONING>), the audit section (wrapped in <AUDIT> and </AUDIT>), AND the final document (wrapped in <DOCUMENT> and </DOCUMENT> tags).
**NEGATIVE CONSTRAINT:** Do NOT repeat the input instructions or any conversational text. Start your response immediately with the <REASONING> tag.

<REASONING>
    [Content from Section IV]
</REASONING>
<AUDIT>
    [Content from Section V]
</AUDIT>
<DOCUMENT>
    [Full text of the Document, formatted as "{doc_type_name}"]
</DOCUMENT>

### Input Context:

**High Recall Query:** {high_recall_query}
**Your Assigned Fact:** {assigned_fact}
**Evidence Blueprint:**
{evidence_blueprint}
```

