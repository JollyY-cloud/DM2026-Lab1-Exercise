import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_pattern_tools


def test_mine_patterns_tool_fixture_fpgrowth():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Known Answer #17: variance and term_frequency keep zero terms
    res_var = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "variance", "algorithm": "fpgrowth", "min_sup": 1})
    assert res_var["patterns"] == []
    assert "no terms survived" in res_var.get("note", "").lower()

    res_tf = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "term_frequency", "algorithm": "fpgrowth", "min_sup": 1})
    assert res_tf["patterns"] == []

    # Known Answer #17: tfidf + fpgrowth
    # catA -> {alpha} support 3
    res_catA = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # catB -> {gamma} support 4
    res_catB = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_tool_fixture_topk_and_maxfpgrowth():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_pattern_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Known Answer #20: algorithm="topk"
    res_topk_a = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": 1})
    assert res_topk_a["n_patterns"] == 1
    assert res_topk_a["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_a["patterns"][0]["support"] == 3

    res_topk_b = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "topk", "k": 1})
    assert res_topk_b["n_patterns"] == 1
    assert res_topk_b["patterns"][0]["pattern"] == ["gamma"]
    assert res_topk_b["patterns"][0]["support"] == 4

    # Known Answer #20: algorithm="maxfpgrowth" returns 0 patterns
    res_max_a = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": 1})
    assert res_max_a["patterns"] == []

    res_max_b = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": 1})
    assert res_max_b["patterns"] == []
