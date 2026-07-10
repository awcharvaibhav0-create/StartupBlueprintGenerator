"""
agents.py — Multi-Agent Architecture for Startup Blueprint Generator.

Agents are logical units with focused prompt scopes, all coordinated by
AgentOrchestrator which executes exactly TWO IBM Granite LLM calls per
blueprint request (as required by the optimised two-LLM architecture).

Agent roster
------------
1. StartupValidationAgent        — validates idea viability
2. MarketResearchAgent           — market sizing, competitors, SWOT
3. BusinessModelCanvasAgent      — BMC + revenue / pricing model
4. FinancialPlanningAgent        — costs, break-even (prompt slice)
5. FundingGovernmentSchemeAgent  — funding paths + govt schemes
6. InvestorPitchAgent            — investor-ready pitch narrative
7. StartupRoadmapAgent           — phased roadmap + risk + future scope

AgentOrchestrator
-----------------
• Builds LLM Call-1 prompt  (agents 1-4 combined)
• Runs LLM Call-2 prompt    (agents 5-7 + python calcs + RAG context)
• Returns a structured dict consumed by app.py / utils.py

JSON Architecture
-----------------
Both LLM calls return strict JSON only.
LLM Call 1 keys: startup_validation, problem_statement, solution_description,
  target_customer_analysis, competitor_analysis, swot_analysis,
  market_opportunity_analysis, business_model_canvas, revenue_model, pricing_strategy
LLM Call 2 keys: cost_estimation, break_even_analysis, funding_recommendations,
  government_schemes, investor_pitch, startup_roadmap, risk_assessment,
  executive_summary, future_scope, final_recommendations
"""

import os
import json
import logging
from typing import Optional
from urllib import response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBM watsonx.ai client (lazy singleton)
# ---------------------------------------------------------------------------
_wx_model = None


def _get_model():
    """Return a lazily-initialised ibm-watsonx-ai ModelInference instance."""
    global _wx_model
    if _wx_model is not None:
        return _wx_model

    from dotenv import load_dotenv
    load_dotenv()

    api_key    = os.getenv("IBM_API_KEY", "")
    project_id = os.getenv("IBM_PROJECT_ID", "")
    wx_url     = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    model_id   = os.getenv("MODEL_ID", "ibm/granite-4-h-small")

    if not api_key or not project_id:
        raise EnvironmentError(
            "IBM_API_KEY and IBM_PROJECT_ID must be set in your .env file."
        )

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        creds = Credentials(url=wx_url, api_key=api_key)

        # Granite 4 models (e.g. ibm/granite-4-h-small) are chat-only.
        # They use the /ml/v1/text/chat endpoint and accept TextChatParameters
        # (max_tokens).  Passing GenTextParamsMetaNames (max_new_tokens,
        # decoding_method) to a chat model causes the SDK to silently discard
        # them because _prepare_chat_payload validates against TextChatParameters.
        if "granite-4" in model_id.lower():
            from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
            init_params = TextChatParameters(max_tokens=4960)
        else:
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
            init_params = {
                Params.DECODING_METHOD: "greedy",
                Params.MAX_NEW_TOKENS: 4960,
            }

        _wx_model = ModelInference(
            model_id=model_id,
            credentials=creds,
            project_id=project_id,
            params=init_params,
        )
        logger.info("IBM watsonx.ai model initialised: %s", model_id)
        return _wx_model
    except Exception as exc:
        logger.error("Failed to initialise IBM watsonx.ai model: %s", exc)
        raise


def _call_llm(prompt: str) -> str:
    """Send a prompt to IBM Granite and return the generated text string.

    IBM Granite 4 models (e.g. ibm/granite-4-h-small) are chat-only models.
    They must be called via model.chat() using the /ml/v1/text/chat endpoint.
    Calling generate_text() on them hits the text-generation endpoint which
    returns HTTP 200 with an empty generated_text — that is the root cause of
    the empty-response bug.

    Routing logic
    -------------
    • If the model id contains "granite-4" (chat-only family) → use chat().
      Response shape: {"choices": [{"message": {"content": "..."}}]}
      Extracted field: choices[0]["message"]["content"]
    • All other models → use generate_text() which returns a plain str.
    """
    model = _get_model()
    model_id: str = os.getenv("MODEL_ID", "ibm/granite-4-h-small")

    try:
        # ----------------------------------------------------------------
        # Granite 4 family: chat-only model — must use chat() API
        # ----------------------------------------------------------------
        if "granite-4" in model_id.lower():
            messages = [{"role": "user", "content": prompt}]
            response = model.chat(messages=messages)

            logger.info("=" * 80)
            logger.info("RAW IBM CHAT RESPONSE TYPE: %s", type(response))
            logger.info("RAW IBM CHAT RESPONSE:")
            logger.info(repr(response))
            logger.info("=" * 80)

            try:
                text = response["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as exc:
                logger.error(
                    "_call_llm: failed to extract chat content from response: %s — %s",
                    repr(response), exc,
                )
                text = ""

        # ----------------------------------------------------------------
        # All other models: text-generation API
        # ----------------------------------------------------------------
        else:
            response = model.generate_text(prompt=prompt)

            

            # Shape 1: plain string (ibm-watsonx-ai >= 1.0 generate_text default)
            if isinstance(response, str):
                text = response.strip()

            # Shape 2: dict with 'results' list (raw API response / some SDK versions)
            elif isinstance(response, dict):
                try:
                    text = response["results"][0]["generated_text"].strip()
                except (KeyError, IndexError, TypeError):
                    text = str(response).strip()
                logger.debug("_call_llm: received dict response, extracted generated_text")

            # Shape 3: SDK response object with .generated_text attribute
            elif hasattr(response, "generated_text"):
                text = response.generated_text.strip()

            # Shape 4: unknown — coerce and log
            else:
                text = str(response).strip()
                logger.warning(
                    "_call_llm: unexpected response type %s; coerced to str",
                    type(response).__name__,
                )

        logger.debug("_call_llm: response length = %d chars", len(text))
        if not text:
            logger.warning("_call_llm: IBM Granite returned an empty response")
        return text

    except Exception as exc:
        logger.error("IBM Granite LLM call failed: %s", exc)
        raise


# ===========================================================================
# Required JSON keys (used for validation and fallback population)
# ===========================================================================

CALL1_REQUIRED_KEYS: list[str] = [
    "startup_validation",
    "problem_statement",
    "solution_description",
    "target_customer_analysis",
    "competitor_analysis",
    "swot_analysis",
    "market_opportunity_analysis",
    "business_model_canvas",
    "revenue_model",
    "pricing_strategy",
]

CALL2_REQUIRED_KEYS: list[str] = [
    "cost_estimation",
    "break_even_analysis",
    "funding_recommendations",
    "government_schemes",
    "investor_pitch",
    "startup_roadmap",
    "risk_assessment",
    "executive_summary",
    "future_scope",
    "final_recommendations",
]

# Canonical ordered union of all 20 blueprint keys.
# This is the single source of truth used by merge_blueprint_sections(),
# _run_generation(), and any downstream consumer that needs the full list.
ALL_REQUIRED_KEYS: list[str] = CALL1_REQUIRED_KEYS + CALL2_REQUIRED_KEYS


def merge_blueprint_sections(
    call1_sections: Optional[dict],
    call2_sections: Optional[dict],
) -> dict[str, str]:
    """
    Merge LLM Call 1 and Call 2 section dicts into a single guaranteed-complete
    dict containing all 20 required blueprint keys.

    Rules
    -----
    • Call 2 values shadow Call 1 values for any key that appears in both
      (should not happen in normal operation, but is safe).
    • Any key missing from either source dict is added with value "" so
      downstream code can always do sections["key"] without a KeyError.
    • Accepts None for either argument (treats it as an empty dict) so callers
      never need to guard before calling.
    • Extra keys returned by the model beyond the required 20 are preserved.

    Returns a dict[str, str] that always contains every key in ALL_REQUIRED_KEYS.
    """
    merged: dict[str, str] = {}

    # Start with Call 1 content
    for k, v in (call1_sections or {}).items():
        merged[k] = v if isinstance(v, str) else str(v) if v else ""

    # Overlay Call 2 content (non-empty values win; never overwrite with "")
    for k, v in (call2_sections or {}).items():
        v_str = v if isinstance(v, str) else str(v) if v else ""
        if v_str:               # only overlay when there is actual content
            merged[k] = v_str
        elif k not in merged:   # absent in both — still add the key
            merged[k] = ""

    # Guarantee every required key is present, even if both dicts were empty
    for k in ALL_REQUIRED_KEYS:
        if k not in merged:
            merged[k] = ""

    return merged

# ===========================================================================
# Individual Agent prompt builders
# Each agent returns a *prompt fragment* string (not a full LLM call).
# The orchestrator stitches all fragments into one JSON-requesting prompt.
# ===========================================================================

class StartupValidationAgent:
    """Validates the startup idea and defines problem/solution."""

    @staticmethod
    def build_prompt_section(idea: str, industry: str, stage: str) -> str:
        return (
            f"IDEA: {idea}\n"
            f"INDUSTRY: {industry}\n"
            f"STAGE: {stage}\n"
        )


class MarketResearchAgent:
    """Produces competitor analysis, SWOT, and market opportunity."""

    @staticmethod
    def build_prompt_section(idea: str, industry: str, target_market: str) -> str:
        return (
            f"TARGET MARKET: {target_market}\n"
        )


class BusinessModelCanvasAgent:
    """Generates the Business Model Canvas and revenue/pricing strategy."""

    @staticmethod
    def build_prompt_section(idea: str, industry: str, revenue_model: str) -> str:
        return (
            f"PREFERRED REVENUE MODEL: {revenue_model}\n"
        )


class FinancialPlanningAgent:
    """Provides a high-level cost and investment overview (used in LLM Call 1)."""

    @staticmethod
    def build_prompt_section(idea: str, team_size: int, monthly_burn: float) -> str:
        return (
            f"TEAM SIZE: {team_size}\n"
            f"MONTHLY BURN RATE: Rs {monthly_burn:,.0f}\n"
        )


class FundingGovernmentSchemeAgent:
    """Recommends funding sources and relevant government schemes."""

    @staticmethod
    def build_prompt_section(
        idea: str, industry: str, stage: str, rag_context: str
    ) -> str:
        # Keep RAG context but cap it to avoid bloating the prompt
        rag_snippet = rag_context[:1200] if rag_context else ""
        return (
            f"STARTUP STAGE: {stage}\n"
            + (f"\nKNOWLEDGE BASE:\n{rag_snippet}\n" if rag_snippet else "")
        )


class InvestorPitchAgent:
    """Creates an investor-ready pitch narrative."""

    @staticmethod
    def build_prompt_section(idea: str, industry: str, summary_data: str) -> str:
        return (
            f"BLUEPRINT SUMMARY:\n{summary_data}\n"
        )


class StartupRoadmapAgent:
    """Generates phased roadmap, risk assessment, executive summary, and future scope."""

    @staticmethod
    def build_prompt_section(idea: str, industry: str, summary_data: str) -> str:
        # summary_data already included by InvestorPitchAgent — no need to repeat
        return ""


# ===========================================================================
# Agent Orchestrator
# ===========================================================================

class AgentOrchestrator:
    """
    Coordinates all agents and executes exactly two IBM Granite LLM calls.

    Parameters accepted by run():
      idea          — startup idea text
      industry      — industry / sector
      stage         — current startup stage
      target_market — target market description
      revenue_model — preferred revenue model
      team_size     — number of team members
      monthly_burn  — estimated monthly burn rate (Rs)
      rag_context   — pre-retrieved RAG context string
      calc_summary  — formatted string of Python-calculated financials

    Returns a dict with keys:
      llm1_raw, llm2_raw, call1_sections, call2_sections

    call1_sections — dict with CALL1_REQUIRED_KEYS (parsed from JSON)
    call2_sections — dict with CALL2_REQUIRED_KEYS (parsed from JSON)
    """

    def run(
        self,
        idea: str,
        industry: str,
        stage: str,
        target_market: str,
        revenue_model: str,
        team_size: int,
        monthly_burn: float,
        rag_context: str,
        calc_summary: str,
    ) -> dict:
        logger.info("AgentOrchestrator: starting blueprint generation")

        # ---------------------------------------------------------------
        # LLM CALL 1 — Validation + Market Research + BMC + Fin Overview
        # ---------------------------------------------------------------
        prompt1 = self._build_prompt1(
            idea, industry, stage, target_market, revenue_model, team_size, monthly_burn
        )
        logger.info("Executing LLM Call 1 …")
        llm1_raw = _call_llm(prompt1)
        logger.info("LLM Call 1 complete (%d chars)", len(llm1_raw))

        # ---------------------------------------------------------------
        # LLM CALL 2 — Funding + Pitch + Roadmap (uses LLM1 + calcs + RAG)
        # ---------------------------------------------------------------
        # Parse LLM1 output and build a capped summary for LLM2 context
        call1_sections = _parse_json_response(llm1_raw, CALL1_REQUIRED_KEYS, call_num=1)
        # Cap each value at 400 chars to keep LLM2 prompt within token budget
        _CAP = 400
        summary_data = (
            "--- LLM Call 1 Blueprint Data ---\n"
            + "\n".join(
                f"{k.replace('_', ' ').title()}: {v[:_CAP]}{'…' if len(v) > _CAP else ''}"
                for k, v in call1_sections.items() if v
            )
            + f"\n\n--- Python Calculations ---\n{calc_summary}"
        )
        prompt2 = self._build_prompt2(
            idea, industry, stage, rag_context, summary_data
        )
        logger.info("Executing LLM Call 2 …")
        llm2_raw = _call_llm(prompt2)
        logger.info("LLM Call 2 complete (%d chars)", len(llm2_raw))

        # ---------------------------------------------------------------
        # Parse LLM2 output into JSON dict
        # ---------------------------------------------------------------
        call2_sections = _parse_json_response(llm2_raw, CALL2_REQUIRED_KEYS, call_num=2)

        # Build the guaranteed-complete merged dict here so every downstream
        # consumer (app.py, utils.py) reads from one place.
        sections = merge_blueprint_sections(call1_sections, call2_sections)

        missing = [k for k in ALL_REQUIRED_KEYS if not sections.get(k)]
        if missing:
            logger.warning(
                "Blueprint incomplete after both LLM calls — "
                "%d/%d required keys are empty: %s",
                len(missing), len(ALL_REQUIRED_KEYS), missing,
            )
        else:
            logger.info(
                "Blueprint complete — all %d required sections present.",
                len(ALL_REQUIRED_KEYS),
            )

        return {
            "llm1_raw": llm1_raw,
            "llm2_raw": llm2_raw,
            "call1_sections": call1_sections,   # kept for backward compat
            "call2_sections": call2_sections,   # kept for backward compat
            "sections": sections,               # canonical merged 20-key dict
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt1(
        idea, industry, stage, target_market, revenue_model, team_size, monthly_burn
    ) -> str:
        """
        Build LLM Call 1 prompt.

        Design decisions
        ----------------
        - Role line is concise; output rule is stated once and unambiguously.
        - Required keys are listed as a manifest (not a pre-filled skeleton) so
          the model cannot mistake placeholder text for real output values.
        - Value descriptions are given in a separate numbered list, not embedded
          inside the JSON skeleton, so the model writes its own content rather
          than echoing the description strings back.
        - Prompt ends with the bare opening brace `{` so the model is forced to
          generate every key and value from scratch — it cannot just re-emit a
          template it was handed.
        """
        s1 = StartupValidationAgent.build_prompt_section(idea, industry, stage)
        s2 = MarketResearchAgent.build_prompt_section(idea, industry, target_market)
        s3 = BusinessModelCanvasAgent.build_prompt_section(idea, industry, revenue_model)
        s4 = FinancialPlanningAgent.build_prompt_section(idea, team_size, monthly_burn)
        context = s1 + s2 + s3 + s4

        # Key manifest — lists expected keys and what each value should contain.
        # Kept separate from the JSON so the model writes values, not templates.
        key_manifest = (
            "Required keys and their expected content:\n"
            "1. startup_validation      — viability assessment of the idea\n"
            "2. problem_statement       — the core problem being solved\n"
            "3. solution_description    — proposed solution and unique selling point\n"
            "4. target_customer_analysis — customer segments and detailed personas\n"
            "5. competitor_analysis     — key competitors and differentiators\n"
            "6. swot_analysis           — Strengths, Weaknesses, Opportunities, Threats\n"
            "7. market_opportunity_analysis — market size, trends, and growth potential\n"
            "8. business_model_canvas   — all nine Business Model Canvas blocks\n"
            "9. revenue_model           — revenue streams and monetisation approach\n"
            "10. pricing_strategy       — pricing tiers and rationale\n"
        )

        return (
            "You are a startup business analyst.\n"
            "Write a detailed startup analysis as a single JSON object.\n"
            "Rules: output raw JSON only — no markdown, no code fences, "
            "no explanation, no text before or after the JSON.\n"
            "Every value must be a plain string with at least 3 sentences of detail.\n\n"
            "STARTUP CONTEXT:\n"
            + context
            + "\n"
            + key_manifest
            + "\nOutput the JSON object now:\n{"
        )

    @staticmethod
    def _build_prompt2(
        idea, industry, stage, rag_context, summary_data
    ) -> str:
        """
        Build LLM Call 2 prompt.

        Design decisions
        ----------------
        - Mirrors _build_prompt1: concise role, output rule stated once.
        - Required keys listed as a numbered manifest, not embedded in a
          pre-filled skeleton, so the model cannot echo placeholder strings.
        - Prompt ends with bare `{` — forces the model to generate content.
        """
        s5 = FundingGovernmentSchemeAgent.build_prompt_section(
            idea, industry, stage, rag_context
        )
        s6 = InvestorPitchAgent.build_prompt_section(idea, industry, summary_data)
        # StartupRoadmapAgent returns "" — context already carried by s6
        context = s5 + s6

        key_manifest = (
            "Required keys and their expected content:\n"
            "1. cost_estimation          — detailed monthly and one-time cost breakdown\n"
            "2. break_even_analysis      — break-even timeline, unit count, and assumptions\n"
            "3. funding_recommendations  — seed, angel, VC, and debt options with rationale\n"
            "4. government_schemes       — relevant Indian government grants and schemes\n"
            "5. investor_pitch           — compelling investor narrative and funding ask\n"
            "6. startup_roadmap          — phased milestones: 0-3, 3-6, 6-12, 12-24 months\n"
            "7. risk_assessment          — top risks and mitigation strategies\n"
            "8. executive_summary        — concise overview for stakeholders\n"
            "9. future_scope             — long-term vision, expansion, and product evolution\n"
            "10. final_recommendations   — prioritised action items for the founding team\n"
        )

        return (
            "You are a startup advisor and investor pitch coach.\n"
            "Write a detailed startup funding and growth plan as a single JSON object.\n"
            "Rules: output raw JSON only — no markdown, no code fences, "
            "no explanation, no text before or after the JSON.\n"
            "Every value must be a plain string with at least 3 sentences of detail.\n\n"
            "CONTEXT:\n"
            + context
            + "\n"
            + key_manifest
            + "\nOutput the JSON object now:\n{"
        )


# ===========================================================================
# JSON parser — replaces the old regex Markdown section parser
# ===========================================================================

import re as _re

# Compiled once at module load — used by _extract_json_from_text.
_FENCE_CLOSED   = _re.compile(r"```(?:json)?\s*([\s\S]*?)```", _re.IGNORECASE)
_FENCE_OPEN     = _re.compile(r"```(?:json)?\s*([\s\S]+)",     _re.IGNORECASE)
_SINGLE_Q_KEY   = _re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'(\s*:)")
_SINGLE_Q_VAL   = _re.compile(r"([:,\[{]\s*)'([^'\\]*(?:\\.[^'\\]*)*)'")


def _normalise_single_quotes(text: str) -> str:
    """
    Best-effort conversion of a single-quoted JSON-like string to double-quoted.

    Only applied when the text starts with `{` and `json.loads` has already
    failed — it is never run on valid JSON.  The approach: replace `'key':` and
    `value'` patterns with double-quote equivalents, then re-check parseability.
    Not perfect for pathological inputs, but recovers the common model artefact.
    """
    # Replace keys:  'word':  →  "word":
    out = _SINGLE_Q_KEY.sub(r'"\1"\2', text)
    # Replace values after  :  [  {  ,  →  "..."
    out = _SINGLE_Q_VAL.sub(lambda m: m.group(1) + '"' + m.group(2) + '"', out)
    return out


def _salvage_truncated(fragment: str) -> str:
    """
    Attempt to close a truncated JSON object so it becomes parseable.

    Handles three truncation shapes that arise when the model exhausts its
    token budget mid-generation:

      (a) Truncated mid-string value  — e.g. {"a": "complete", "b": "incomplet
      (b) Truncated after comma       — e.g. {"a": "complete", "b": "done",
      (c) Truncated mid-key name      — e.g. {"a": "complete", "b": "done", "ke

    Strategy
    --------
    1.  Replay a character scanner to learn the final parser state:
        - Was the scanner inside a string when the input ended?
        - Which containers (braces/brackets) are still open?
        - What was the byte position of the last *safely completed* key-value
          pair (i.e. just before any dangling comma or partial token)?
    2.  Build repair candidates from least to most destructive:
        Attempt A — close open string (if any) + close containers.
        Attempt B — strip dangling comma/partial-key token back to the last
                    safe position, then close containers.
    3.  Return the first candidate that round-trips through json.loads, or
        the original fragment if none work (caller validates).
    """
    start = fragment.find("{")
    if start == -1:
        return fragment
    s = fragment[start:]

    # ── Phase 1: replay character scanner ──────────────────────────────────
    depth            = 0
    in_string        = False
    escape_next      = False
    container_stack: list[str] = []   # '{' or '[' pushed on open, popped on close

    # Track the index of the last character that completed a value at depth 1
    # (i.e. just after a closing `"` or `}` or `]` or digit-run that is a
    # complete value).  We use this as a safe rewind point for case (c).
    last_safe_pos    = 0   # index into s[], 0 = just the opening `{`

    for idx, ch in enumerate(s):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            was_in = in_string
            in_string = not in_string
            # Closing a string at depth 1: this is the end of either a key or
            # a value.  We can only know which by context, but recording the
            # position after the closing quote is safe enough — if it turns
            # out to be after a key, the rewind will strip the dangling `:`.
            if was_in and depth == 1:
                last_safe_pos = idx        # position of the closing `"`
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            container_stack.append(ch)
            depth += 1
        elif ch in ("}", "]"):
            if container_stack:
                container_stack.pop()
            depth -= 1
            if depth == 1:
                last_safe_pos = idx        # safe after closing a nested object
        elif ch == "," and depth == 1:
            # A comma at depth 1 separates key-value pairs.  The position
            # *before* the comma is the last safe end of a complete pair.
            last_safe_pos = idx - 1

    # ── Phase 2: build closing suffix ──────────────────────────────────────
    def _close(base: str) -> str:
        """Append closing chars for every open container (innermost first)."""
        for opener in reversed(container_stack):
            base += "}" if opener == "{" else "]"
        return base

    # Attempt A — close open string then close containers
    base_a = s
    if in_string:
        base_a = s + '"'         # close the dangling open string
    candidate_a = _close(base_a.rstrip())

    # Attempt B — rewind to last_safe_pos, strip trailing comma/colon/ws,
    # then close containers.  This removes dangling partial-key tokens and
    # trailing commas that would make the JSON invalid.
    safe_fragment = s[:last_safe_pos + 1].rstrip()
    # Strip a trailing colon left over if last_safe_pos was after a key string
    if safe_fragment.endswith(":"):
        safe_fragment = safe_fragment[:-1].rstrip()
    # Strip a trailing comma
    if safe_fragment.endswith(","):
        safe_fragment = safe_fragment[:-1].rstrip()
    candidate_b = _close(safe_fragment)

    # Return the first candidate that is valid JSON; otherwise return original.
    for c in (candidate_a, candidate_b):
        try:
            json.loads(c)
            return c
        except json.JSONDecodeError:
            pass
    return fragment


def _extract_json_from_text(text: str) -> str:
    """
    Robustly extract a JSON object from raw LLM output.

    Strategies tried in order — first success wins:

    0.  Prepend `{` — handles responses where the model continued after our
        prompt-priming brace (body arrives without the leading `{`).
    1.  Direct parse of the stripped text — fastest path for well-formed output.
    2a. Closed code fence  ```json … ```  — strip wrappers and parse interior.
    2b. Unclosed code fence  ```json …   — strip opening fence, parse remainder.
    3.  Balanced-brace scan — walks character-by-character tracking string
        context and brace depth; extracts the outermost complete `{ … }` even
        when trailing prose follows it.
    4.  Naive first-`{` to last-`}` slice — fast fallback for simple cases.
    5.  Truncation salvage — closes open strings and containers so a response
        truncated mid-token can still be partially parsed.
    6.  Single-quote normalisation — converts `'key': 'val'` style output to
        valid double-quoted JSON and retries strategies 1 and 3.

    Returns the best JSON string found, or the original stripped text if all
    strategies fail (so the caller can log the raw content).
    """
    stripped = text.strip()
    if not stripped:
        return stripped

    def _try(candidate: str, tag: str) -> str | None:
        """Parse candidate; return it if valid, else None."""
        try:
            json.loads(candidate)
            logger.debug("_extract_json_from_text: %s succeeded", tag)
            return candidate
        except json.JSONDecodeError:
            return None

    # ── Strategy 0: prepend missing opening brace ────────────────────────
    # Fires when the model continued after our priming `{` and returned the
    # body without the leading brace.  Two sub-cases:
    #   0a — body ends with `}` (most common when prompt ends with `{`)
    #   0b — body starts with `"key":` (model dropped both braces entirely)
    if not stripped.startswith("{"):
        # 0a: has closing brace — just prepend
        if stripped.endswith("}"):
            result = _try("{" + stripped, "S0a prepend-{")
            if result:
                return result
        # 0b: starts with a JSON key string — wrap in braces
        if stripped.startswith('"') or stripped.startswith("'"):
            result = _try("{" + stripped + "}", "S0b wrap-braces")
            if result:
                return result
            # also try without trailing brace in case model already ended one
            result = _try("{" + stripped, "S0b prepend-{ only")
            if result:
                return result

    # ── Strategy 1: direct parse ─────────────────────────────────────────
    result = _try(stripped, "S1 direct")
    if result:
        return result

    # ── Strategy 2a: closed code fence ───────────────────────────────────
    m = _FENCE_CLOSED.search(stripped)
    if m:
        result = _try(m.group(1).strip(), "S2a closed-fence")
        if result:
            return result

    # ── Strategy 2b: unclosed code fence (model truncated inside fence) ──
    m = _FENCE_OPEN.search(stripped)
    if m:
        result = _try(m.group(1).strip(), "S2b open-fence")
        if result:
            return result

    # ── Strategy 3: balanced-brace scan ──────────────────────────────────
    first_brace = stripped.find("{")
    if first_brace != -1:
        depth       = 0
        in_string   = False
        escape_next = False
        end_pos     = -1
        for i, ch in enumerate(stripped[first_brace:], start=first_brace):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        if end_pos != -1:
            result = _try(stripped[first_brace:end_pos + 1], "S3 brace-scan")
            if result:
                return result

    # ── Strategy 4: naive first/last-brace slice ──────────────────────────
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        result = _try(stripped[first_brace:last_brace + 1], "S4 naive-slice")
        if result:
            return result

    # ── Strategy 5: truncation salvage ───────────────────────────────────
    # Pick the best fragment to repair: if there is a code fence, work on its
    # interior; otherwise work on everything from the first `{`.
    fence_interior: str | None = None
    m = _FENCE_OPEN.search(stripped)
    if m:
        fence_interior = m.group(1).strip()
    fragment = fence_interior if fence_interior else (
        stripped[first_brace:] if first_brace != -1 else stripped
    )
    salvaged = _salvage_truncated(fragment)
    if salvaged != fragment:
        result = _try(salvaged, "S5 truncation-salvage")
        if result:
            return result

    # ── Strategy 6: single-quote normalisation ────────────────────────────
    # Try on the raw stripped text, and also on whatever fragment S3 isolated.
    for src, tag in (
        (stripped,                            "S6a single-quote raw"),
        (stripped[first_brace:] if first_brace != -1 else "", "S6b single-quote brace-fragment"),
    ):
        if not src:
            continue
        normalised = _normalise_single_quotes(src)
        if normalised != src:
            result = _try(normalised, tag)
            if result:
                return result
            # Also try balanced-brace scan on the normalised text
            fb = normalised.find("{")
            if fb != -1:
                depth2 = 0; in_str2 = False; esc2 = False; ep2 = -1
                for i, ch in enumerate(normalised[fb:], start=fb):
                    if esc2:             esc2 = False;  continue
                    if ch == "\\" and in_str2: esc2 = True; continue
                    if ch == '"':        in_str2 = not in_str2; continue
                    if in_str2:          continue
                    if ch == "{":        depth2 += 1
                    elif ch == "}":
                        depth2 -= 1
                        if depth2 == 0:  ep2 = i; break
                if ep2 != -1:
                    result = _try(normalised[fb:ep2 + 1], tag + "+brace-scan")
                    if result:
                        return result

    # ── All strategies exhausted ──────────────────────────────────────────
    logger.warning(
        "_extract_json_from_text: all strategies failed; "
        "returning raw text for downstream error logging (len=%d)", len(stripped)
    )
    return stripped


def _coerce_to_str(value: object) -> str:
    """Coerce any JSON value to str without using Python repr for containers."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _parse_json_response(
    text: str,
    required_keys: list[str],
    call_num: int = 0,
) -> dict[str, str]:
    """
    Parse an LLM JSON response into a dict[str, str].

    • Calls _extract_json_from_text to handle every known LLM output shape.
    • If the extracted text is still not valid JSON, logs the error and
      falls back to an empty dict — never raises.
    • Every key in required_keys is guaranteed to be present in the result;
      missing keys get an empty string rather than being silently dropped.
    • Non-string values (nested objects, arrays, numbers, booleans, null)
      are coerced via _coerce_to_str, which uses json.dumps for containers
      so the result is valid JSON text rather than Python repr.
    • Partial responses (from truncation salvage) may contain only a subset
      of keys — those remaining keys are filled with "" and a warning is logged
      per key so the operator can diagnose token-budget issues.
    • Extra keys returned by the model beyond required_keys are preserved.

    Returns a dict[str, str] guaranteed to contain every key in required_keys.
    """
    label = f"LLM Call {call_num}" if call_num else "LLM"
    json_text = _extract_json_from_text(text)
    logger.info("=" * 80)
    logger.info("%s Extracted JSON:", label)
    logger.info(json_text)
    logger.info("=" * 80)

    parsed: dict = {}
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            logger.warning(
                "%s: parsed JSON is not a dict (type=%s); using empty dict.",
                label, type(parsed).__name__,
            )
            parsed = {}
        else:
            recovered = len(parsed)
            missing   = [k for k in required_keys if k not in parsed]
            if missing:
                logger.warning(
                    "%s: JSON parsed with %d/%d required keys present; "
                    "missing: %s",
                    label, recovered, len(required_keys), missing,
                )
            else:
                logger.info(
                    "%s: JSON parsed successfully — all %d required keys present.",
                    label, len(required_keys),
                )
    except json.JSONDecodeError as exc:
        logger.error(
            "%s: JSON parse failed after all extraction strategies (%s). "
            "Raw output (first 500 chars):\n%s",
            label, exc, text[:500],
        )
        parsed = {}

    # ── Validate and normalise required keys ────────────────────────────
    result: dict[str, str] = {}
    for key in required_keys:
        raw = parsed.get(key, "")
        if not isinstance(raw, str):
            logger.warning(
                "%s: key '%s' has type %s, expected str; coercing.",
                label, key, type(raw).__name__,
            )
            raw = _coerce_to_str(raw)
        if not raw:
            logger.warning(
                "%s: key '%s' is missing or empty in JSON response.", label, key
            )
        result[key] = raw

    # ── Preserve bonus keys the model returned ───────────────────────────
    for key, value in parsed.items():
        if key not in result:
            result[key] = _coerce_to_str(value)

    return result


def get_section(sections: dict, *keys: str) -> str:
    """
    Retrieve a section value from the JSON blueprint dict.

    Lookup order:
    1. Exact key match (case-insensitive, underscore/space normalised).
    2. Substring match against all keys.

    Returns the first match or an empty string.
    """
    def _normalise(s: str) -> str:
        return s.lower().replace(" ", "_").replace("-", "_")

    norm_map = {_normalise(k): v for k, v in sections.items()}

    for key in keys:
        nk = _normalise(key)
        # exact
        if nk in norm_map:
            return norm_map[nk]
        # substring
        for k, v in norm_map.items():
            if nk in k or k in nk:
                return v
    return ""
