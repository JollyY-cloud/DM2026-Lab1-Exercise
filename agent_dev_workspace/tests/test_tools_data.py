import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_fixture():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student tools
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    # Known Answer #2 assertions from TEST_FIXTURE.md
    assert res["total_rows"] == 8
    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]
    assert res["missing_percentage"] == pytest.approx(12.5, abs=1e-3)


def test_check_duplicates_tool_inspection():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Known Answer #3 inspection (drop=False)
    res = check_duplicates_tool.invoke({"drop": False})

    assert res["duplicate_indices"] == [5, 6]
    assert res["n_duplicates"] == 2
    assert res["remaining_rows"] == 8
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_dropping():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Known Answer #3 dropping (drop=True)
    res = check_duplicates_tool.invoke({"drop": True})

    assert res["action_taken"] == "dropped_all_copies"
    assert res["remaining_rows"] == 6
    assert len(session.dataframe) == 6
    assert 5 not in session.dataframe.index
    assert 6 not in session.dataframe.index


def test_tokenize_tool_fixture():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Run tokenize_tool
    res = tokenize_tool.invoke({})

    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27  # 5 + 4 + 3 + 5 + 4 + 3 + 3 + 0 = 27

    # Verify unigrams column in session.dataframe
    assert "unigrams" in session.dataframe.columns
    unigrams_df = session.dataframe["unigrams"].tolist()

    # Verify exact token lists from Known Answer #10 in TEST_FIXTURE.md
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []  # Row 7 empty list for empty text
    ]

    assert unigrams_df == expected_unigrams
    assert session.artifacts.get("unigrams") == expected_unigrams


def test_sample_data_tool_fixture():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Known Answer #13 assertions (n=4, random_state=42)
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["n_requested"] == 4
    assert res["n_sampled"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(res["sampled_data"]) == 4

    # Confirm original session.dataframe remains unchanged
    assert len(session.dataframe) == 8
    assert session.dataframe.index.tolist() == list(range(8))

    # Test error handling for invalid n
    res_err = sample_data_tool.invoke({"n": 0})
    assert res_err.get("status") == "error"


def test_describe_data_tool_fixture():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # Run describe_data_tool
    res = describe_data_tool.invoke({})

    # Known Answer #18 text lengths: [29, 23, 17, 30, 24, 18, 18, 0]
    assert "text_length" in session.dataframe.columns
    assert session.dataframe["text_length"].tolist() == [29, 23, 17, 30, 24, 18, 18, 0]

    # Verify overall describe stats from Known Answer #18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # Verify catA stats from Known Answer #18
    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29

    # Verify catB stats from Known Answer #18
    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30

    # Verify boxplot figure created on session
    assert session.pending_figure is not None


def test_list_files_tool_fixture():
    session = SessionState()

    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known Answer #11 assertions for subdirectory="newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Test inspect_data_tool with n=3
    res = inspect_data_tool.invoke({"n": 3})

    assert res["total_rows"] == 8
    assert res["n_preview"] == 3
    assert len(res["preview"]) == 3

    # Check actual columns returned by session.dataframe
    expected_cols = session.dataframe.columns.tolist()
    assert res["columns"] == expected_cols

    # Check dtypes map matches actual session.dataframe dtypes
    for col in expected_cols:
        assert col in res["dtypes"]
        assert res["dtypes"][col] == str(session.dataframe[col].dtype)

    # Test error handling for invalid n
    res_err = inspect_data_tool.invoke({"n": -1})
    assert res_err.get("status") == "error"
