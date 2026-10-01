# Single Positive Document Generation

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `SINGLE_POSITIVE_DOCUMENT_PROMPTS`
- Runtime generator: [`data_gen/pos_gen.py`](../data_gen/pos_gen.py)
- Active version: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

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

You are a knowledge base curator tasked with generating **exactly 1 Independent Positive Document** to support the provided High Recall Query. Your *only* source of truth for content is the **Evidence Blueprint** provided below.

### II. Disentanglement Constraints

The document MUST strictly adhere to the following:
1. **Atomic Fact Embedding:** The document MUST contain the **EXACT STRING** of the fact/data point assigned to it (see "Your Assigned Fact" below). This string must be present character-for-character.
2. **Data Disentanglement:** This document must NOT contain information from other documents' blueprint entries. It must be impossible to answer the full query using only this document.
3. **Implicit Retrieval (Contextual Framing):** Frame the required fact/data point within a dense, natural language context (e.g., an excerpt from a technical brief or news article)."
4. **Independence & Length:** The document must be a unique block of text (at least {word_count} words) that can stand alone.

### Feedback from Previous Attempts (If Any)
{feedback}

### III. Planning & Reasoning (The Blueprint)

**Output Requirement:** You must use the strict XML structure defined below. First, plan the content of the document to ensure it meets the disentanglement constraints.

    <REASONING>
        <BLUEPRINT_ANALYSIS>
            [Analyze the Evidence Blueprint and confirm assignment of facts to the document]
        </BLUEPRINT_ANALYSIS>
        <DISENTANGLEMENT_PLAN>
            [Explain how you will ensure that the document doesn't reveal the full answer]
        </DISENTANGLEMENT_PLAN>
    </REASONING>

### IV. Mandatory Audit & Self-Correction

**Output Requirement:** You must audit the generated document against the constraints. You must **always** output the full <AUDIT> block. 

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
        <OBSERVATION>[Is it impossible to answer the query with just one document?]</OBSERVATION>
    </CHECK_DISENTANGLEMENT>
</AUDIT>

### V. Final Output Generation

After completing your reasoning and mandatory audit, provide the final document.

**IMPORTANT:** You must include ALL three sections: the reasoning section (wrapped in <REASONING> and </REASONING>), the audit section (wrapped in <AUDIT> and </AUDIT>), AND the final document (wrapped in <DOCUMENT> and </DOCUMENT> tags).
**NEGATIVE CONSTRAINT:** Do NOT repeat the input instructions or any conversational text. Start your response immediately with the <REASONING> tag.

<REASONING>
    [Content from Section III]
</REASONING>
<AUDIT>
    [Content from Section IV]
</AUDIT>
<DOCUMENT>
    [Full text of the Document]
</DOCUMENT>

### Input Context:

**High Recall Query:** {high_recall_query}
**Your Assigned Fact:** {assigned_fact}
**Evidence Blueprint:**
{evidence_blueprint}
```

