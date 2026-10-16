import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool_fixture():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer #16 assertions from TEST_FIXTURE.md
    # 1. Doc 0 vs Doc 1 (both catA, near-identical): 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 2. Doc 0 vs Doc 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    # 3. Doc 3 vs Doc 4 (both catB, near-identical): 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 4. Doc 0 vs Doc 7 (catA vs empty-text row): 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)

    # Test error handling for out-of-range doc index
    res_err = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 99})
    assert res_err.get("status") == "error"


def test_feature_correlation_matrix_tool_fixture():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Run feature_correlation_matrix_tool
    res = feature_corr_tool.invoke({})

    assert res["n_features"] == 5

    # Known Answer #19 top terms order by variance descending
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms_by_variance"] == expected_terms

    corr_mat = res["correlation_matrix"]
    assert len(corr_mat) == 5
    assert len(corr_mat[0]) == 5

    # Helper to check symmetric correlation value between term1 and term2
    def get_corr(t1, t2):
        i1 = expected_terms.index(t1)
        i2 = expected_terms.index(t2)
        return corr_mat[i1][i2]

    # Known Answer #19 correlation matrix assertions
    assert get_corr("alpha", "alpha") == pytest.approx(1.0000, abs=1e-3)
    assert get_corr("alpha", "gamma") == pytest.approx(-0.5718, abs=1e-3)
    assert get_corr("alpha", "delta") == pytest.approx(-0.6882, abs=1e-3)
    assert get_corr("alpha", "beta") == pytest.approx(0.8885, abs=1e-3)
    assert get_corr("alpha", "always") == pytest.approx(0.2601, abs=1e-3)

    assert get_corr("gamma", "gamma") == pytest.approx(1.0000, abs=1e-3)
    assert get_corr("gamma", "delta") == pytest.approx(0.8307, abs=1e-3)
    assert get_corr("gamma", "beta") == pytest.approx(-0.6435, abs=1e-3)
    assert get_corr("gamma", "always") == pytest.approx(0.3140, abs=1e-3)

    assert get_corr("delta", "delta") == pytest.approx(1.0000, abs=1e-3)
    assert get_corr("delta", "beta") == pytest.approx(-0.7746, abs=1e-3)
    assert get_corr("delta", "always") == pytest.approx(0.3780, abs=1e-3)

    assert get_corr("beta", "beta") == pytest.approx(1.0000, abs=1e-3)
    assert get_corr("beta", "always") == pytest.approx(0.2928, abs=1e-3)

    assert get_corr("always", "always") == pytest.approx(1.0000, abs=1e-3)

    # Verify heatmap figure assigned to pending_figure
    assert session.pending_figure is not None
