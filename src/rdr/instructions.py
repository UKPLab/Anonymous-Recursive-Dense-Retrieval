"""Instruction strings, as used by the paper's evaluation code (they produced the paper's numbers).

Table 7 of the paper prints them with small editorial differences (CoIR spelling) and prints another string
family for BRIGHT; the constants below are the operative strings.

Update instructions drive every context-bearing level (h >= 2); root instructions follow each
benchmark's own convention (including its spelling). Use them by key::

    from rdr import instructions
    instructions.get("continuation")
    instructions.root_for("nanobeir", "NanoSciFact")
"""

from __future__ import annotations

from typing import Dict, Optional

# ----------------------------------------------------------------- update instructions (h >= 2)
RECOVERY = (
    "The previously retrieved document is a distractor or partial observation. Given the query and this "
    "document, retrieve the document that fully satisfies the information need."
)
"""Recovery (single-hop): relevance refinement. Drives the primary single-hop tree."""

CONTINUATION = (
    "Given the question and the documents already retrieved, retrieve the next document needed to "
    "continue answering it."
)
"""Continuation: evidence completion. Drives the multi-hop tree and BRIGHT-Pro."""

RECOVERY_MULTIHOP = (
    "The previously retrieved document is a distractor. Given the question and this incorrect document, "
    "retrieve the correct document that continues the chain."
)
"""Chain recovery: the optional recovery view of multi-hop trees (Appendix E.1)."""

# ----------------------------------------------------------------------- root instructions
MULTIHOP_ROOT = "Retrieve the first document needed to begin answering this multi-hop question."
"""Shared root instruction of MuSiQue, BrowseComp+, FanOutQA and FRAMES."""

# BRIGHT: the strings of the paper's single-hop evaluation code, which produced the BRIGHT numbers of Tables 2,
# 5, 24–29 (verbatim, including "a Earth Science"). Table 7 of the paper prints the "Represent this {domain} post
# for searching relevant passages:" family instead; with those strings the RDR-8B root collapses on several
# StackExchange domains.
_BRIGHT_SE = "Given a {domain} post, retrieve relevant passages that help answer the post"
BRIGHT_ROOTS: Dict[str, str] = {
    "biology": _BRIGHT_SE.format(domain="Biology"),
    "earth_science": _BRIGHT_SE.format(domain="Earth Science"),
    "economics": _BRIGHT_SE.format(domain="Economics"),
    "psychology": _BRIGHT_SE.format(domain="Psychology"),
    "robotics": _BRIGHT_SE.format(domain="Robotics"),
    "stackoverflow": _BRIGHT_SE.format(domain="Stack Overflow"),
    "sustainable_living": _BRIGHT_SE.format(domain="Sustainable Living"),
    "pony": "Given a Pony coding problem, retrieve relevant passages that help answer the problem",
    "leetcode": "Given a coding problem, retrieve relevant examples that help answer the problem",
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem",
}

NANOBEIR_ROOTS: Dict[str, str] = {
    "NanoArguAna": "Given a claim, find documents that refute the claim",
    "NanoClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "NanoDBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "NanoFEVER": "Given a claim, retrieve documents that support or refute the claim",
    "NanoFiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "NanoHotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "NanoMSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NanoNFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NanoNQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "NanoQuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "NanoSCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "NanoSciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "NanoTouche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
}

# CoIR: the strings the paper's evaluation code uses (the official CoIR/MTEB instructions, including their
# spelling, e.g. "retrieval code"). Table 7 of the paper prints a cleaned-up version.
_CODE_Q = "Given a question about coding, retrieval code or passage that can solve user's question"
COIR_ROOTS: Dict[str, str] = {
    "codetrans-dl": "Given a piece for code, retrieval semantically similar code",
    "codetrans-contest": "Given a piece for code, retrieval semantically similar code",
    "CodeSearchNet-ccr": "Given a code comment, retrieve the code snippet corresponding to that comment.",
    "CodeSearchNet": "Given a code snippet, retrieve the comment corresponding to that code.",
    "apps": "Given a question about code problem, retrieval code that can solve user's problem",
    "cosqa": _CODE_Q,
    "stackoverflow-qa": _CODE_Q,
    "codefeedback-st": _CODE_Q,
    "codefeedback-mt": _CODE_Q,
    "synthetic-text2sql": "Given a user's question, retrieve SQL queries that are appropriate responses to the question",
}

TOOLRET_FALLBACK = "Retrieve the most relevant tool."
"""ToolRet uses each query's own instruction from the dataset; this is the fallback."""

TOPIOCQA_ROOT = "Given a conversation, retrieve the next turn"
BRIGHT_PRO_ROOT = "Instruct: Given a {task} post, retrieve relevant passages that help answer the post"

INSTRUCTIONS: Dict[str, str] = {
    "recovery": RECOVERY,
    "continuation": CONTINUATION,
    "recovery_multihop": RECOVERY_MULTIHOP,
    "multihop": MULTIHOP_ROOT,
    "topiocqa": TOPIOCQA_ROOT,
    "toolret": TOOLRET_FALLBACK,
    "bright_pro": BRIGHT_PRO_ROOT,
}


def get(key: str) -> str:
    """Instruction by key (``recovery``, ``continuation``, ``recovery_multihop``, ``multihop`` …).

    Args:
        key: One of :data:`INSTRUCTIONS`.
    """
    try:
        return INSTRUCTIONS[key]
    except KeyError as e:
        raise KeyError(f"unknown instruction key {key!r}; known: {sorted(INSTRUCTIONS)}") from e


def root_for(benchmark: str, task: Optional[str] = None) -> str:
    """The paper's root instruction for a benchmark/task, e.g. ``root_for("bright", "biology")``.

    Args:
        benchmark: ``musique``, ``browsecomp``, ``fanoutqa``, ``frames``, ``bright``, ``nanobeir``, ``coir``,
            ``topiocqa``, ``toolret`` or ``bright_pro``.
        task: Task within the benchmark (BRIGHT domain, NanoBEIR dataset, CoIR task, BRIGHT-Pro task).
    """
    b = benchmark.lower()
    if b in ("musique", "browsecomp", "browsecomp_plus", "browsecomp+", "fanoutqa", "frames", "multihop"):
        return MULTIHOP_ROOT
    if b == "bright":
        t = (task or "").replace("Bright-", "")
        return BRIGHT_ROOTS[t]
    if b == "nanobeir":
        return NANOBEIR_ROOTS[task if task.startswith("Nano") else "Nano" + task]
    if b == "coir":
        t = task or ""
        if t.startswith("CodeSearchNet-ccr"):
            return COIR_ROOTS["CodeSearchNet-ccr"]
        if t.startswith("CodeSearchNet"):
            return COIR_ROOTS["CodeSearchNet"]
        return COIR_ROOTS[t]
    if b in ("topiocqa", "chat"):
        return TOPIOCQA_ROOT
    if b == "toolret":
        return TOOLRET_FALLBACK
    if b in ("bright_pro", "bright-pro"):
        return BRIGHT_PRO_ROOT.format(task=task or "")
    raise KeyError(f"no root instruction for benchmark {benchmark!r}")
