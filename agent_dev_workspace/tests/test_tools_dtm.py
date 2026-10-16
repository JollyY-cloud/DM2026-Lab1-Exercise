import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_fixture():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student tool using the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})

    # Known Answer #4 assertions from TEST_FIXTURE.md
    assert res["n_documents"] == 8
    assert res["n_features"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.artifacts.get("count_vectorizer") is not None


def test_build_dtm_tool_invalid_ngram_range():
    session = SessionState()
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Call on empty session should return error
    res_no_data = build_dtm_tool.invoke({})
    assert res_no_data.get("status") == "error"

    # Load dataset and test invalid ngram_range
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res_invalid_ngram = build_dtm_tool.invoke({"ngram_range": [2, 1]})
    assert res_invalid_ngram.get("status") == "error"


def test_term_frequency_tool_fixture():
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
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Run term_frequency_tool
    res = term_frequency_tool.invoke({})

    assert res["n_terms"] == 5
    assert res["total_term_occurrences"] == 27  # 6 + 7 + 3 + 4 + 7 = 27

    # Check full_report stored in results_store
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]
    assert full_report is not None
    assert list(full_report.columns[:2]) == ["term", "frequency"]

    # Verify exact frequencies from Known Answer #14
    freq_dict = dict(zip(full_report["term"], full_report["frequency"]))
    assert freq_dict["alpha"] == 6
    assert freq_dict["always"] == 7
    assert freq_dict["beta"] == 3
    assert freq_dict["delta"] == 4
    assert freq_dict["gamma"] == 7


def test_dtm_heatmap_tool_fixture():
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
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Test full matrix default slice (20x20) matching Known Answer #21
    res_full = dtm_heatmap_tool.invoke({})

    assert res_full["n_documents"] == 8
    assert res_full["n_terms"] == 5
    assert res_full["included_terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_full["doc_labels"] == [f"Doc {i}" for i in range(8)]
    
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test sliced call (n_terms=2, n_documents=3)
    session.pending_figure = None  # reset figure
    res_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res_sliced["n_documents"] == 3
    assert res_sliced["n_terms"] == 2
    assert res_sliced["included_terms"] == ["alpha", "always"]
    assert res_sliced["doc_labels"] == ["Doc 0", "Doc 1", "Doc 2"]
    assert res_sliced["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
