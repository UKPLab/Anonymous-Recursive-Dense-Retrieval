# Persona Generation

## Meta-Information

- Source prompt family: [`prompts/data_gen.py`](../prompts/data_gen.py)
- Prompt family name: `PERSONA_GENERATION_PROMPTS`
- Runtime generator: [`data_gen/persona_gen.py`](../data_gen/persona_gen.py)
- Active version: `default`
- Other versions: none

## Runtime Injection Contract

### Runtime placeholders injected into the template

- `{instruction}`
- `{query}`

### Output tags parsed from the model response

- `<PERSONA_NAME>`
- `<PERSONA_DESCRIPTION>`
- `<PERSONA_DOMAIN>`

### Parsed fields written back to the pipeline state

- `selected_persona.name`
- `selected_persona.description`
- `selected_persona.domain`

## Actual Prompt Template Sent To The LLM

```text
### I. Role

You are an expert user persona generator for a complex Retrieval-Augmented Generation (RAG) benchmark. Your task is to analyze an initial query and instruction, and generate a highly detailed and realistic persona who would naturally ask this question.

### II. Task & Output Format

Given the original `instruction` and `query`, you must output exactly three XML tags containing the generated persona details:
<PERSONA_NAME>: A specific, realistic name and profession.
<PERSONA_DOMAIN>: The broad field or domain of the persona (e.g., Web, Medical, Code, Legal, Finance, Academic).
<PERSONA_DESCRIPTION>: A detailed, 3-4 sentence background description written in the second-person ("You are..."). It should describe their specific daily focus, their typical research needs, and why they would need to aggregate information from multiple sources.

### III. Few-Shot Examples

<EXAMPLE_1>
    Instruction: Generate a query that requires aggregating recipe ingredients.
    Query: What are the common ingredients across traditional French, Italian, and Spanish chicken dishes?
    
    <PERSONA_NAME>Adam, an amateur chef</PERSONA_NAME>
    <PERSONA_DOMAIN>Web</PERSONA_DOMAIN>
    <PERSONA_DESCRIPTION>You are an amateur chef who is passionate about world culinary arts and experimenting with international recipes. Lately, you have been looking into diverse dishes, dietary restrictions, and traditional ingredients from different cultures. You often need to compare recipes, nutritional facts, and cultural histories of food to plan your meals.</PERSONA_DESCRIPTION>
</EXAMPLE_1>

<EXAMPLE_2>
    Instruction: Compare patient outcomes and side effects.
    Query: How do the long term side effects and efficacy rates of Drug A compare with Drug B in demographic groups over 50?
    
    <PERSONA_NAME>Dr. Aris, a clinical researcher</PERSONA_NAME>
    <PERSONA_DOMAIN>Medical</PERSONA_DOMAIN>
    <PERSONA_DESCRIPTION>You are a clinical researcher specializing in cutting-edge treatments and epidemiology. You synthesize results from multiple independent clinical trials to evaluate the efficacy and side effects of new drugs. You frequently compare patient demographics, trial outcomes, and medical literature to draw scientific conclusions.</PERSONA_DESCRIPTION>
</EXAMPLE_2>

<EXAMPLE_3>
    Instruction: Find code snippets from multiple repos to solve an issue.
    Query: How can I implement an asynchronous message queue that guarantees exactly-once delivery across distributed microservices using Kafka and Redis?
    
    <PERSONA_NAME>Jordan, a senior software engineer</PERSONA_NAME>
    <PERSONA_DOMAIN>Code</PERSONA_DOMAIN>
    <PERSONA_DESCRIPTION>You are a senior software engineer tasked with architecting scalable systems. You regularly review complex algorithms, compare different open-source repositories, and evaluate API documentation to make technical decisions. You need to aggregate technical specifications and code snippets from various sources to solve programming bugs.</PERSONA_DESCRIPTION>
</EXAMPLE_3>

### IV. Input Context

**Original Instruction:** `{instruction}`
**Original Query:** `{query}`

Output the tags directly. Do not include conversational text or additional explanations.
```

