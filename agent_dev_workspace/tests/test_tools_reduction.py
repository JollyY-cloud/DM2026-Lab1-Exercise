import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca_fixture():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #7: PCA
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2, "random_state": 42})

    assert res["method"] == "pca"
    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)

    # Verify explained_variance_ratio [0.7532, 0.1965]
    exp_var = res["explained_variance_ratio"]
    assert exp_var[0] == pytest.approx(0.7532, abs=1e-3)
    assert exp_var[1] == pytest.approx(0.1965, abs=1e-3)

    # Expected PCA coordinates from Known Answer #7
    expected_pca_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    # Compare per component column allowing for consistent sign flips per component
    for col in range(2):
        diff_direct = np.abs(coords[:, col] - expected_pca_coords[:, col])
        diff_flipped = np.abs(coords[:, col] + expected_pca_coords[:, col])
        assert np.all(diff_direct < 0.05) or np.all(diff_flipped < 0.05)

    # Verify scatter plot figure created on session
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne_fixture():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #8: t-SNE with perplexity=2, random_state=42
    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })

    assert res["method"] == "tsne"
    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)

    # Verify scatter plot figure created on session
    assert session.pending_figure is not None


def test_reduce_dimensions_umap_fixture():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #9: UMAP with n_neighbors=3, random_state=42
    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    assert res["method"] == "umap"
    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)

    # Verify scatter plot figure created on session
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Record original session state attributes before invoking binarize_labels_tool
    orig_labels = list(session.labels) if session.labels is not None else None
    orig_df_len = len(session.dataframe)

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # Run binarize_labels_tool
    res = binarize_labels_tool.invoke({"column_name": "category_name"})

    assert res["n_samples"] == 8
    assert res["categories"] == ["catA", "catB"]
    assert res["n_categories"] == 2

    # Expected 8x2 one-hot matrix (catA: [1, 0], catB: [0, 1])
    expected_one_hot = [
        [1, 0],  # row 0: catA
        [1, 0],  # row 1: catA
        [1, 0],  # row 2: catA
        [0, 1],  # row 3: catB
        [0, 1],  # row 4: catB
        [0, 1],  # row 5: catB
        [0, 1],  # row 6: catB
        [1, 0]   # row 7: catA
    ]

    assert res["one_hot_matrix"] == expected_one_hot
    assert session.artifacts.get("binarized_labels") == expected_one_hot
    assert session.artifacts.get("binarized_categories") == ["catA", "catB"]

    # Verify session.labels and session.dataframe remain unmodified
    if orig_labels is not None:
        assert list(session.labels) == orig_labels
    assert len(session.dataframe) == orig_df_len
