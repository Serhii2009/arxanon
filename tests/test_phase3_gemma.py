"""Phase 3 Gemma validator tests: check_ollama, extract_formal_structure, validate_bridge_pair."""
import pytest

from arxanon.clusters import BridgePair, BridgePipelineResult, TDAResult


# ── Shared mock helpers ───────────────────────────────────────────────────────

_STRUCTURE = {
    "problem_type": "stability_analysis",
    "key_relationships": "dV/dt <= -alpha*V(x)",
    "solution_approach": "Lyapunov method",
    "assumptions": "bounded domain",
    "domain_vocabulary": "energy function",
}


def _fake_tool_call(arguments: dict):
    """Return a minimal tool-call chain: response.message.tool_calls[0].function.arguments."""
    fn = type("Fn", (), {"arguments": arguments})()
    tc = type("TC", (), {"function": fn})()
    return tc


def _fake_response(arguments: dict, thinking: str = "", content: str = ""):
    msg = type("Msg", (), {
        "tool_calls": [_fake_tool_call(arguments)],
        "thinking": thinking,
        "content": content,
    })()
    return type("Resp", (), {"message": msg})()


# ── check_ollama ──────────────────────────────────────────────────────────────

def test_check_ollama_connection_error(monkeypatch):
    """Unreachable host → (False, error_message), never raises."""
    import requests as req

    monkeypatch.setattr(req, "get", lambda *a, **k: (_ for _ in ()).throw(
        req.exceptions.ConnectionError("refused")
    ))

    from arxanon.gemma_validator import check_ollama
    ok, msg = check_ollama(base_url="http://bad-host:9999", model="gemma4:27b")

    assert not ok
    assert msg is not None
    assert len(msg) > 0


def test_check_ollama_connection_error_via_side_effect(monkeypatch):
    """Alternative mock approach for ConnectionError."""
    import requests as req

    def _raise(*a, **k):
        raise req.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(req, "get", _raise)

    from arxanon.gemma_validator import check_ollama
    ok, msg = check_ollama(base_url="http://bad-host:9999", model="gemma4:27b")

    assert not ok
    assert "not running" in msg or "bad-host" in msg


def test_check_ollama_model_not_found(monkeypatch):
    """Ollama running but target model not pulled → (False, pull instruction)."""
    import requests as req

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]}

    monkeypatch.setattr(req, "get", lambda *a, **k: _Resp())

    from arxanon.gemma_validator import check_ollama
    ok, msg = check_ollama(model="gemma4:27b")

    assert not ok
    assert msg is not None


def test_check_ollama_success(monkeypatch):
    """Model present in Ollama tags → (True, None)."""
    import requests as req

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "gemma4:27b"}, {"name": "llama3:8b"}]}

    monkeypatch.setattr(req, "get", lambda *a, **k: _Resp())

    from arxanon.gemma_validator import check_ollama
    ok, msg = check_ollama(model="gemma4:27b")

    assert ok is True
    assert msg is None


def test_check_ollama_model_prefix_match(monkeypatch):
    """'gemma4:27b-instruct-q4' satisfies check for model base 'gemma4'."""
    import requests as req

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "gemma4:27b-instruct-q4_K_M"}]}

    monkeypatch.setattr(req, "get", lambda *a, **k: _Resp())

    from arxanon.gemma_validator import check_ollama
    ok, _ = check_ollama(model="gemma4:27b")
    assert ok


# ── extract_formal_structure ──────────────────────────────────────────────────

def test_extract_structure_returns_dict(monkeypatch):
    """Mocked ollama.Client.chat with tool_call → returns parsed structure dict."""
    import ollama as ollama_lib

    resp = _fake_response(_STRUCTURE)
    monkeypatch.setattr(ollama_lib.Client, "chat", lambda self, *a, **k: resp)

    from arxanon.gemma_validator import extract_formal_structure
    result = extract_formal_structure("An abstract about Lyapunov stability.")

    assert result is not None
    assert result["problem_type"] == "stability_analysis"
    assert result["solution_approach"] == "Lyapunov method"


def test_extract_structure_no_tool_call(monkeypatch):
    """ollama returns a response with no tool_calls → None."""
    import ollama as ollama_lib

    msg = type("Msg", (), {"tool_calls": None})()
    resp = type("Resp", (), {"message": msg})()
    monkeypatch.setattr(ollama_lib.Client, "chat", lambda self, *a, **k: resp)

    from arxanon.gemma_validator import extract_formal_structure
    result = extract_formal_structure("Some abstract.")

    assert result is None


def test_extract_structure_ollama_exception(monkeypatch):
    """ollama.chat raises → None is returned, no exception propagates."""
    import ollama as ollama_lib

    def _boom(self, *a, **k):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(ollama_lib.Client, "chat", _boom)

    from arxanon.gemma_validator import extract_formal_structure
    result = extract_formal_structure("Some abstract.")

    assert result is None


# ── validate_bridge_pair ──────────────────────────────────────────────────────

_PA = {"arxiv_id": "cs001", "abstract": "Abstract A: stability via Lyapunov.", "categories": '["cs.LG"]'}
_PB = {"arxiv_id": "math001", "abstract": "Abstract B: energy methods for PDEs.", "categories": '["math.DS"]'}


def test_validate_pair_superficial_returns_none(monkeypatch):
    """SUPERFICIAL classification → validate_bridge_pair returns None."""
    import arxanon.gemma_validator as gv

    monkeypatch.setattr(gv, "extract_formal_structure", lambda *a, **k: _STRUCTURE)

    classify_args = {
        "classification": "SUPERFICIAL",
        "reasoning": "No real structural match found.",
        "matched_properties": [],
        "mismatched_properties": ["everything"],
    }
    monkeypatch.setattr(
        gv, "_chat_with_think_fallback",
        lambda *a, **k: (_fake_response(classify_args, thinking=""), False),
    )

    result = gv.validate_bridge_pair(_PA, _PB, sim_score=0.75)
    assert result is None


def test_validate_pair_structural_high_confidence(monkeypatch):
    """STRUCTURAL classification with ≥5 matched properties → BridgePair(HIGH)."""
    import arxanon.gemma_validator as gv

    monkeypatch.setattr(gv, "extract_formal_structure", lambda *a, **k: _STRUCTURE)

    classify_args = {
        "classification": "STRUCTURAL",
        "reasoning": "Same stability framework in different domains.",
        "matched_properties": ["stability", "convergence", "energy", "boundedness", "invariance"],
        "mismatched_properties": ["domain vocabulary"],
        "translation_hints": [{"term_a": "energy", "term_b": "Lyapunov function"}],
    }
    monkeypatch.setattr(
        gv, "_chat_with_think_fallback",
        lambda *a, **k: (
            _fake_response(classify_args, thinking="Deep structural analogy confirmed."),
            True,
        ),
    )

    result = gv.validate_bridge_pair(_PA, _PB, sim_score=0.92)

    assert result is not None
    assert isinstance(result, BridgePair)
    assert result.classification == "STRUCTURAL"
    assert result.confidence_tier == "HIGH"
    assert result.think_mode_used is True
    assert len(result.matched_properties) == 5
    assert result.reasoning_chain  # non-empty guarantee


def test_validate_pair_reasoning_chain_fallback(monkeypatch):
    """When think=False is used, reasoning_chain falls back to args['reasoning']."""
    import arxanon.gemma_validator as gv

    monkeypatch.setattr(gv, "extract_formal_structure", lambda *a, **k: _STRUCTURE)

    classify_args = {
        "classification": "METHODOLOGICAL",
        "reasoning": "Both use iterative fixed-point methods.",
        "matched_properties": ["iteration", "convergence", "fixed_point"],
        "mismatched_properties": ["notation"],
    }
    resp = _fake_response(classify_args, thinking="", content="")  # no thinking
    monkeypatch.setattr(
        gv, "_chat_with_think_fallback",
        lambda *a, **k: (resp, False),
    )

    result = gv.validate_bridge_pair(_PA, _PB, sim_score=0.80)

    assert result is not None
    assert result.think_mode_used is False
    assert result.reasoning_chain  # must be non-empty even without think mode


def test_validate_pair_extraction_failure_returns_none(monkeypatch):
    """extract_formal_structure returning None → validate_bridge_pair returns None."""
    import arxanon.gemma_validator as gv

    monkeypatch.setattr(gv, "extract_formal_structure", lambda *a, **k: None)

    result = gv.validate_bridge_pair(_PA, _PB, sim_score=0.85)
    assert result is None


def test_validate_pair_no_tool_calls_returns_none(monkeypatch):
    """Classification response with no tool_calls → None."""
    import arxanon.gemma_validator as gv

    monkeypatch.setattr(gv, "extract_formal_structure", lambda *a, **k: _STRUCTURE)

    msg = type("Msg", (), {"tool_calls": None, "thinking": None, "content": ""})()
    resp = type("Resp", (), {"message": msg})()
    monkeypatch.setattr(
        gv, "_chat_with_think_fallback",
        lambda *a, **k: (resp, False),
    )

    result = gv.validate_bridge_pair(_PA, _PB, sim_score=0.80)
    assert result is None


# ── validate_clusters (via run_gemma_validation) ──────────────────────────────

def test_validate_clusters_gemma_unavailable(monkeypatch):
    """Ollama unavailable → gemma_available=False, clusters unchanged, no crash."""
    import arxanon.gemma_validator as gv
    monkeypatch.setattr(gv, "check_ollama", lambda *a, **k: (False, "Ollama not running at localhost"))

    from arxanon.bridge_pipeline import run_gemma_validation

    original_result = BridgePipelineResult(
        graph_nodes=10,
        graph_edges_full=20,
        graph_edges_bridge=15,
        bibcoupling_edges_added=0,
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[]),
        clusters=[],
    )

    updated = run_gemma_validation(original_result, papers={}, top_n_clusters=5)

    assert updated.gemma_available is False
    assert updated.gemma_warning is not None
    assert updated.clusters == []  # unchanged
