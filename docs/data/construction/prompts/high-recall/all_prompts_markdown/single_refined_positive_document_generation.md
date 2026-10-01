# Single Refined Positive Document Generation

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `SINGLE_REFINED_POSITIVE_DOCUMENT_PROMPTS`
- Runtime generator: [`data_gen/refined_pos_gen.py`](../data_gen/refined_pos_gen.py)
- Active version: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{word_count}`
- `{feedback}`
- `{few_shot_examples}`
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

You are a knowledge base curator tasked with generating **exactly 1 Independent Positive Document** to support the provided High Recall Query. Your *only* source of truth for factual content is the detailed **Evidence Blueprint** provided below.

### II. Disentanglement & Stylistic Constraints

The document MUST strictly adhere to the following:
1. **Atomic Fact Embedding:** The document MUST contain the **EXACT STRING** of the fact/data point assigned to it (see "Your Assigned Fact" below). This string must be present character-for-character to allow for automated verification.
2. **Data Disentanglement:** This document must NOT contain information from other documents' blueprint entries. It must be impossible to answer the full query using only this document.
3. **Implicit Retrieval:** Frame the required fact within a dense, natural language context. Avoid explicitly stating "This is the answer" or something similarly blatant.
4. **Mimetic Style Transfer:** Use the provided stylized example (in Section V) as a template for tone, vocabulary, and structural formatting (e.g., news snippets, technical memos). However, DO NOT use any factual data from the example. Only use your assigned fact and generate matching contextual details.
5. **Independence & Length:** The document must be a unique block of text (at least {word_count} words) that can stand alone.

### Feedback from Previous Attempts (If Any)
{feedback}

### III. Planning & Reasoning

**Output Requirement:** You must use the strict XML structure defined below. First, plan the content of the document.

    <REASONING>
        <BLUEPRINT_ANALYSIS>
            [Confirm your assigned fact from the blueprint and check what shouldn't be included]
        </BLUEPRINT_ANALYSIS>
        <STYLE_ALIGNMENT_PLAN>
            [How will you mimic the style example without leaking its facts?]
        </STYLE_ALIGNMENT_PLAN>
        <DISENTANGLEMENT_PLAN>
            [Explain how you will ensure the document remains independent and doesn't reveal the full query answer]
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
    <CHECK_STYLE_ADHERENCE>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Does the document strongly mimic the tone and structure of the stylistic example?]</OBSERVATION>
    </CHECK_STYLE_ADHERENCE>
</AUDIT>

### V. Stylistic Archetype
Use the following example only to guide the structure, tone, and formatting of your output. **Do not copy facts from it.**

{few_shot_examples}

### VI. Final Output Generation

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
    [Full text of your Document]
</DOCUMENT>

### Input Context:

**High Recall Query:** {high_recall_query}
**Your Assigned Fact:** {assigned_fact}
**Evidence Blueprint:**
{evidence_blueprint}
```

