# Eval Judge

## Meta-Information

- Source prompt family: [`prompts/eval.py`](../prompts/eval.py)
- Prompt family name: `EVAL_JUDGE_PROMPTS`
- Runtime generators: [`evaluation/judge_gen.py`](../evaluation/judge_gen.py)
- Active version: `2.0.0`
- Other versions present in source: `default`

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{query}`
- `{reference_answer}`
- `{predicted_answer}`

### Output tags parsed from the model response

- `<REASONING>`
- `<VERDICT>`

### Parsed fields written back to the pipeline state

- `eval_pos_neg_verdict`
- `eval_pos_neg_reasoning`
- `eval_pos_only_verdict`
- `eval_pos_only_reasoning`

## Actual Prompt Template Sent To The LLM

```text
### Role
You are an expert evaluator for Question Answering systems.

### Task
Compare the <PREDICTED_ANSWER> with the <REFERENCE_ANSWER> for the given <QUERY>.
Determine if the PREDICTED_ANSWER is semantically equivalent to the REFERENCE_ANSWER, or contains the correct core information required by the REFERENCE_ANSWER.

### Inputs
<QUERY>
{query}
</QUERY>

<REFERENCE_ANSWER>
{reference_answer}
</REFERENCE_ANSWER>

<PREDICTED_ANSWER>
{predicted_answer}
</PREDICTED_ANSWER>

### Rules
1.  Focus on the factual content.
2.  Allow for differences in phrasing, length, or formatting.
3.  If the predicted answer says "Insufficient information" but the reference answer has an answer, it is INCORRECT.
4.  If the predicted answer contains the correct information but adds extra irrelevant details, it is CORRECT (unless it contradicts itself).

### Output Format
You must format your response exactly as follows, with no additional text before or after:

<REASONING>
    [Step-by-step verification. Identify specific missing evidence or confirm the presence of facts.]
</REASONING>

<VERDICT>
    [CORRECT | INCORRECT]
</VERDICT>
```

