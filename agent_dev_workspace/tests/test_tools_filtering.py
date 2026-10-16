import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_fixture():
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

    # Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold 0.15 matching Known Answer #5
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["n_original_features"] == 5
    assert res["n_kept_features"] == 4
    assert res["n_removed_features"] == 1
    assert res["kept_features"] == ["alpha", "beta", "delta", "gamma"]
    assert res["removed_features"] == ["always"]

    # Check full_report stored in results_store for visualize_result_tool
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]
    assert full_report is not None
    assert list(full_report.columns[:2]) == ["term", "variance"]
    
    # Verify exact individual term variances from Known Answer #5
    var_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-3)


def test_pearson_filter_tool_fixture():
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

    # Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run pearson_filter_tool for target_class="catB" matching Known Answer #6
    res = pearson_filter_tool.invoke({"target_class": "catB"})

    assert res["target_class"] == "catB"
    assert res["n_features"] == 5

    # Check full_report stored in results_store
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]
    assert full_report is not None
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Verify exact Pearson correlation values from Known Answer #6
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)


def test_spearman_filter_tool_fixture():
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

    # Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run spearman_filter_tool for target_class="catB" matching Known Answer #6
    res = spearman_filter_tool.invoke({"target_class": "catB"})

    assert res["target_class"] == "catB"
    assert res["n_features"] == 5

    # Check full_report stored in results_store
    stored = session.results_store[res["result_id"]]
    full_report = stored["full_report"]
    assert full_report is not None
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Verify exact Spearman correlation values from Known Answer #6
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
