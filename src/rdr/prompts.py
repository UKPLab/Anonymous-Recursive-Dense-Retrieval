"""State rendering: ``s = [iota; q; ctx(O_s)]`` (Equation 1).

A :class:`PromptConfig` decides which instruction each role uses and how a state is laid out as
text. The renderer produces the complete string that is encoded; the encoder adds nothing.

Instruction *wrap*:

* ``"replace"`` — context-bearing states put the update instruction in the ``Instruct:`` slot
  (the trained format; the paper's multi-hop protocol)::

      Instruct: {continuation}
      Query:{question}
      Document 1: {d1}
      Document 2: {d2}

* ``"keep"`` — context-bearing states keep the benchmark instruction and add the update instruction
  after the query (the paper's single-hop protocol)::

      Instruct: {benchmark instruction}
      Query: {query}
      {recovery}
      Retrieved Context:
      Document 1: {d1}

The observation block can differ per level (the single-hop tree renders hop 2 as
``Retrieved Context:`` + ``Document 1:`` and hops 3–4 as ``Incorrect document N:``). The presets
follow the rendering of the paper's evaluation code; every field can be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence

from . import instructions as I
from .types import ASSEMBLY, CONTINUATION, FEEDBACK, RECOVERY, ROOT

RECOVERY_VIEW = "recovery_view"


@dataclass
class ContextLayout:
    """How the observed documents of a state are written.

    Attributes:
        header: Optional line before the documents (e.g. ``"Retrieved Context:"``).
        item: One document, ``{i}`` is 1-based, ``{text}`` the document.
    """

    header: str = ""
    item: str = "Document {i}: {text}"

    def render(self, docs: Sequence[str]) -> str:
        """Render observed documents (``{i}`` counts from 1).

        Args:
            docs: Document texts in path order.
        """
        body = "\n".join(self.item.format(i=i + 1, text=d) for i, d in enumerate(docs))
        return (self.header + "\n" + body) if self.header else body


@dataclass
class PromptConfig:
    """Instructions and text layout of all states.

    Attributes:
        root: Default root instruction; a per-query instruction passed to ``search(..., instruction=...)``
            overrides it (benchmark instructions differ per task).
        instructions: Role -> update instruction. Roles: ``continuation``, ``recovery`` (primary roles),
            ``recovery_view`` (the recovery re-encoding of a continuation tree), ``feedback`` (joint
            pseudo-relevance feedback), ``assembly`` (set-conditioned selection).
        primary_role: Role of the context-bearing states that drive the search.
        wrap: ``"replace"``: the update instruction replaces the root instruction in the head (multi-hop);
            ``"keep"``: the head keeps the root instruction and the update follows the query (single-hop).
        wrap_by_role: Per-role override of ``wrap``.
        head: Instruction/query line, ``{instruction}`` and ``{query}``.
        layouts: Observation layout per number of observed documents (``1``, ``2`` …).
        default_layout: Layout for every other number of observed documents.
        feedback_layout: Layout of the joint-feedback state.
        incorrect: Rendering of the last observation in the multi-hop recovery view (``{text}``).
        strip: Strip whitespace from instruction, query and documents (single-hop protocol).
        observation_max_tokens: Token cap per injected document (multi-hop: 512); ``None`` disables it.
        name: Name of the configuration (presets set it).
    """

    root: Optional[str] = I.MULTIHOP_ROOT
    instructions: Dict[str, str] = field(default_factory=lambda: {
        CONTINUATION: I.CONTINUATION,
        RECOVERY: I.RECOVERY,
        RECOVERY_VIEW: I.RECOVERY_MULTIHOP,
        FEEDBACK: "",
        ASSEMBLY: I.CONTINUATION,
    })
    primary_role: str = CONTINUATION
    wrap: str = "replace"
    wrap_by_role: Dict[str, str] = field(default_factory=dict)
    head: str = "Instruct: {instruction}\nQuery:{query}"
    layouts: Dict[int, ContextLayout] = field(default_factory=dict)
    default_layout: ContextLayout = field(default_factory=ContextLayout)
    feedback_layout: Optional[ContextLayout] = None
    incorrect: str = "Retrieved (incorrect):\n{text}"
    strip: bool = False
    observation_max_tokens: Optional[int] = 512
    name: str = "custom"

    @staticmethod
    def preset(name: str, **overrides) -> "PromptConfig":
        """Named configurations (see :data:`PRESETS`): ``"multi-hop"``, ``"single-hop"``,
        ``"single-hop-continuation"``, ``"bright-pro"``.

        Args:
            name: Preset name.
            **overrides: Fields to replace, e.g. ``PromptConfig.preset("single-hop", root="…")``.
                ``instructions`` is merged into the preset's instructions (only the given roles change).
        """
        key = name.lower().replace("_", "-")
        if key not in PRESETS:
            raise KeyError(f"unknown prompt preset {name!r}; available: {sorted(PRESETS)}")
        cfg = PRESETS[key]()
        if "instructions" in overrides:  # override single roles, keep the preset's other roles
            overrides = {**overrides, "instructions": {**cfg.instructions, **overrides["instructions"]}}
        return replace(cfg, **overrides) if overrides else cfg

    def instruction_for(self, role: str) -> str:
        if role == "primary":
            role = self.primary_role
        try:
            return self.instructions[role]
        except KeyError as e:
            raise KeyError(f"no instruction configured for role {role!r}") from e

    def layout_for(self, n_docs: int) -> ContextLayout:
        return self.layouts.get(n_docs, self.default_layout)

    def as_sentence_transformers_prompts(self) -> Dict[str, str]:
        """The instruction prefixes as a sentence-transformers ``prompts`` dict (root/continuation/...)."""
        out = {ROOT: self.head.format(instruction=self.root or "", query="")}
        for role, instr in self.instructions.items():
            out[role] = self.head.format(instruction=instr, query="")
        return out


class StateRenderer:
    """Renders states to text with a :class:`PromptConfig` and an optional token-level truncation.

    Args:
        config: The prompt configuration.
        truncate: ``truncate(text, max_tokens)`` used for ``observation_max_tokens`` (the retriever passes
            the encoder tokenizer's encode/decode truncation).
    """

    def __init__(self, config: PromptConfig, truncate: Optional[Callable[[str, Optional[int]], str]] = None) -> None:
        self.config = config
        self._truncate = truncate

    def _doc(self, text: str) -> str:
        cfg = self.config
        if cfg.strip:
            text = str(text).strip()
        if cfg.observation_max_tokens and self._truncate is not None:
            text = self._truncate(text, cfg.observation_max_tokens)
        return text

    def _head(self, instruction: str, query: str) -> str:
        if self.config.strip:
            instruction, query = (instruction or "").strip(), (query or "").strip()
        return self.config.head.format(instruction=instruction, query=query)

    def render(self, query: str, observations: Sequence[str], role: str, instruction: Optional[str] = None,
               level: Optional[int] = None) -> str:
        """Render one state.

        Args:
            query: The query (conversations: the serialized dialogue).
            observations: Observed document texts in path order (empty for the root).
            role: ``root`` or an update role (``continuation``, ``recovery``, ``recovery_view``, ``feedback``,
                ``assembly`` …).
            instruction: This query's root instruction (overrides ``config.root``).
            level: Informational (1 = root).
        """
        cfg = self.config
        root_instr = instruction if instruction is not None else (cfg.root or "")
        if role == ROOT or (not observations and role != FEEDBACK):
            # the root, and the first (empty-set) step of Assembly
            return self._head(root_instr, query)
        docs = [self._doc(o) for o in observations]
        if role == FEEDBACK:
            # joint feedback (ANCE-PRF protocol): the root's instruction and query, then all feedback documents
            layout = cfg.feedback_layout or cfg.layout_for(len(docs))
            update = cfg.instructions.get(FEEDBACK, "")
            text = self._head(root_instr, query) + (("\n" + update) if update else "")
            return text + "\n" + layout.render(docs)
        update = cfg.instruction_for(role)
        wrap = cfg.wrap_by_role.get(role, cfg.wrap)
        if role == RECOVERY_VIEW:
            ctx_docs, last = docs[:-1], docs[-1]
        else:
            ctx_docs, last = docs, None
        if wrap == "replace":
            text = self._head(update, query)
        elif wrap == "keep":
            text = self._head(root_instr, query) + (("\n" + update) if update else "")
        else:
            raise ValueError(f"unknown wrap {wrap!r}")
        if ctx_docs:
            text += "\n" + cfg.layout_for(len(ctx_docs)).render(ctx_docs)
        if last is not None:
            text += "\n" + cfg.incorrect.format(text=last)
        return text


# ------------------------------------------------------------------ presets
def _multihop() -> PromptConfig:
    """The paper's multi-hop contract (MuSiQue, BrowseComp+, FanOutQA, FRAMES)."""
    return PromptConfig(root=I.MULTIHOP_ROOT, name="multi-hop")


def _singlehop() -> PromptConfig:
    """The paper's single-hop contract: benchmark root instruction, recovery at every context level."""
    return PromptConfig(
        root=None,
        instructions={RECOVERY: I.RECOVERY, CONTINUATION: I.CONTINUATION, RECOVERY_VIEW: I.RECOVERY,
                      FEEDBACK: "", ASSEMBLY: I.RECOVERY},
        primary_role=RECOVERY,
        wrap="keep",
        head="Instruct: {instruction}\nQuery: {query}",
        layouts={1: ContextLayout(header="Retrieved Context:", item="Document {i}: {text}")},
        default_layout=ContextLayout(header="", item="Incorrect document {i}: {text}"),
        feedback_layout=ContextLayout(header="Retrieved Context:", item="Document {i}: {text}"),
        strip=True,
        observation_max_tokens=None,
        name="single-hop",
    )


def _singlehop_continuation() -> PromptConfig:
    """Single-hop tree driven by the continuation (next-document) instruction (Tables 14, 25, 31)."""
    return replace(_singlehop(), primary_role=CONTINUATION, name="single-hop-continuation")


def _bright_pro() -> PromptConfig:
    """BRIGHT-Pro: continuation at every context level (Table 2)."""
    return replace(_singlehop_continuation(), name="bright-pro")


PRESETS: Dict[str, Callable[[], PromptConfig]] = {
    "multi-hop": _multihop,
    "multihop": _multihop,
    "single-hop": _singlehop,
    "singlehop": _singlehop,
    "single-hop-continuation": _singlehop_continuation,
    "bright-pro": _bright_pro,
}
