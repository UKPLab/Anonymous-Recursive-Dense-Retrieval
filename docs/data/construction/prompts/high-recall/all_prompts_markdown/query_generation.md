# Query Generation

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `QUERY_GENERATION_PROMPTS`
- Runtime generator: [`data_gen/query_gen.py`](../data_gen/query_gen.py)
- Active version: `2.3.0`
- Other versions present in source: `2.2.2`, `2.2.1`, `2.2.0`, `default`, `2.0.0`, `2.1.0`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{instruction}`
- `{query}`
- `{selected_scenario}`
- `{scenario_description}`
- `{num_documents}`
- `{feedback}`
- `{persona_name}`
- `{persona_description}`
- `{persona_domain}`

### Output tags parsed from the model response

- `<QUERY>` or fallback `<FINAL_QUERY>`
- `<REASONING>`
- `<INITIAL_EVIDENCE_BLUEPRINT>`
- `<EVIDENCE_BLUEPRINT>`
- `<AUDIT>`

### Parsed fields written back to the pipeline state

- `high_recall_query`
- `reasoning`
- `initial_evidence_blueprint`
- `evidence_blueprint`
- `audit`
- `selected_scenario`
- `num_documents`

## Actual Prompt Template Sent To The LLM

```text
### I. Role and Persona

You are an expert retrieval benchmark designer specializing in constructing complex, **High Recall Retrieval** queries that test the ability of LLMs to aggregate and synthesize information across multiple sources. 

To ensure the query is highly realistic and natural, you must adopt the following target persona for this task:
**Persona Name:** {persona_name}
**Background:** {persona_description}

You will formulate the query strictly from the perspective of this persona, asking a question they would naturally need answered in their daily work, studies, or research.

### II. High Recall Constraints (The Goal)

Your task is to convert this input into a single, comprehensive question that enforces the following High Recall Scenario: {selected_scenario}.

{scenario_description}

The resulting query must meet these core criteria:
1.  **Multi-Document Synthesis:** Answering the question must require locating and synthesizing information from exactly {num_documents} independent documents**.
2.  **Implicit Retrieval:** The query **must be discreet** and cannot explicitly mention or hint at the type of document, source, or file needed (e.g., avoid terms like 'report', 'document', 'file', 'article', 'web page', 'technical specification', etc.). The need for retrieval must be entirely implicit, such as by asking for a calculation, comparison, or synthesis.
3.  **Parallel Aggregation:** The process must demand parallel aggregation or synthesis, **not** rely on a linear, sequential chain of dependency (multi-hop bias).
4.  **Topical Fidelity:** The reformulated query MUST remain on the **same topic, domain, and entities** as the Original Query and Instruction. The scenario description is only a structural template for HOW to frame the question (e.g., aggregation, comparison) — it must NOT replace the subject matter. For example, if the original query is about a movie actor, the reformulated query must still be about that movie actor, NOT about an unrelated corporation or financial metric.
5.  **Logical Coherence:** The reformulated query must be a meaningful, non-trivial question that makes real-world sense. Do NOT generate queries with trivially obvious or nonsensical answers (e.g., "How has someone's age evolved over time?" is trivially answered by the passage of time and is NOT a valid query).
6.  **Temporal Specificity:** Any time references in the query must use **exact years or dates** (e.g., "from 2018 to 2023", "in Q3 2022"). Do NOT use vague temporal phrases like "now", "present day", "recently", "in the past few years", or "to date". Vague time references make the query impossible to verify.
7.  **Persona Alignment (CRITICAL):** Speak exactly as your assigned persona would. The query must sound like a genuine, natural information-seeking goal from this persona, NOT a mechanical task description. Do not use meta-language like "aggregate the data" or "compare these sources".

### Feedback from Previous Attempts (If Any)
{feedback}

### III. Scenario-Specific Reasoning Steps (The Blueprint)

**Output Requirement:** You must use the strict XML structure defined below. The <INITIAL_EVIDENCE_BLUEPRINT> is the most critical section. It will be used programmatically to generate the actual documents.
**CRITICAL:** The content inside each <DOC_N> tag must be a **concrete, atomic fact** (e.g., a specific dollar amount, a date, a named entity, or a percentage) that will be embedded *verbatim* into the document. Do not use vague descriptions like "information about sales"; instead write "$12.5 million revenue in Q3".
    
    <REASONING>
        <TASK_DEFINITION>
            State the specific {selected_scenario} operation (e.g., aggregation, comparison) required.
        </TASK_DEFINITION>
        <INITIAL_EVIDENCE_BLUEPRINT>
            <DOC_1> [Exact string/atomic fact for Document 1] </DOC_1>
            <DOC_2> [Exact string/atomic fact for Document 2] </DOC_2>
            <DOC_N> [Exact string/atomic fact for Document N] </DOC_N>
        </INITIAL_EVIDENCE_BLUEPRINT>
        <COGNITIVE_JUSTIFICATION>
            Explain how this new question requires a higher-order cognitive skill (such as Analysis or Application) and defeats simple extractive retrieval.
        </COGNITIVE_JUSTIFICATION>
    </REASONING>

### IV. Mandatory Audit & Self-Correction

**Output Requirement:** You must audit the generated query against the constraints. You must **always** output the full <AUDIT> block. 

Use the following logic for the "STATUS" tags:
- If a check passes, output: **COMPLIANT**
- If a check fails, output: **NON-COMPLIANT**

<AUDIT>
    <CHECK_LEAKAGE>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Briefly state if document types/sources were mentioned]</OBSERVATION>
    </CHECK_LEAKAGE>
    <CHECK_HIGH_RECALL>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Briefly state if {num_documents} pieces of evidence are truly required]</OBSERVATION>
    </CHECK_HIGH_RECALL>
    <CHECK_BLUEPRINT_COUNT>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Briefly state if the blueprint contains exactly {num_documents} items]</OBSERVATION>
    </CHECK_BLUEPRINT_COUNT>
    <CHECK_TOPIC>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Does the reformulated query preserve the same topic, domain, and entities as the Original Query and Instruction?]</OBSERVATION>
    </CHECK_TOPIC>  
    <CHECK_LOGICAL_COHERENCE>
        <STATUS>[COMPLIANT or NON-COMPLIANT]</STATUS>
        <OBSERVATION>[Is the query a meaningful, non-trivial question?]</OBSERVATION>
    </CHECK_LOGICAL_COHERENCE>
    <REFINEMENT>
        <ACTION>[NONE or REFINED]</ACTION>
        <REASONING>[If REFINED, state specific fix. If NONE, state N/A]</REASONING>
        <FINAL_QUERY>[Insert the Final, Validated Query Here]</FINAL_QUERY>
    </REFINEMENT>
</AUDIT>

### V. Final Output Generation

After completing your reasoning and mandatory audit, provide the final, fully-formed High Recall Question.

**IMPORTANT:** You must include ALL three sections: the reasoning section (wrapped in <REASONING> and </REASONING>), the audit section (wrapped in <AUDIT> and </AUDIT>), AND the final output (wrapped in <EVIDENCE_BLUEPRINT> and <QUERY> tags).
**NEGATIVE CONSTRAINT:** Do NOT repeat the input instructions, the scenario description, or any conversational text (e.g., "Here is the output"). Start your response immediately with the <REASONING> tag.

<REASONING>
    [Content from Section III]
</REASONING>
<AUDIT>
    [Content from Section IV]
</AUDIT>
<EVIDENCE_BLUEPRINT>
    [Insert the Final Evidence Blueprint here. Steps:
     1. If you refined the query in IV.B, you MUST update the blueprint to match the new query.
     2. If no refinement was needed, copy the <INITIAL_EVIDENCE_BLUEPRINT> from Section III.]
</EVIDENCE_BLUEPRINT>
<QUERY>
    [Insert the final High Recall Retrieval Query here]
</QUERY>

### Input Context:

**Original MAIR Input:**
**Original Instruction:** `{instruction}`
**Original Query:** `{query}`
```

