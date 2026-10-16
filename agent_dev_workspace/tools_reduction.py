from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """
        Reduces Document-Term Matrix dimensionality to 2D using PCA, t-SNE, or UMAP and plots a scatter plot.

        Args:
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap').
            n_components: Number of dimensions for output (default 2).
            random_state: Seed for reproducible random sampling/initialization.
            perplexity: Perplexity parameter for t-SNE (adjusted for dataset size).
            n_neighbors: Neighbor count parameter for UMAP (adjusted for dataset size).

        Returns:
            Dict containing method, 2D coordinates per document, and explained variance ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        X = session.feature_matrix.toarray()
        n_samples, n_features = X.shape

        explained_variance_ratio = None

        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = np.round(model.explained_variance_ratio_, 4).tolist()
        elif method == "tsne":
            # Adjust perplexity if larger than sample count
            actual_perp = min(perplexity, max(1.0, float(n_samples - 1)))
            model = TSNE(
                n_components=n_components,
                perplexity=actual_perp,
                random_state=random_state,
                init="random"
            )
            coords = model.fit_transform(X)
        elif method == "umap":
            # Adjust n_neighbors if larger than sample count
            actual_neighbors = min(n_neighbors, max(2, n_samples - 1))
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=actual_neighbors,
                random_state=random_state
            )
            coords = model.fit_transform(X)
        else:
            return {
                "status": "error",
                "error": f"Invalid method '{method}'. Choose 'pca', 'tsne', or 'umap'."
            }

        # Round coordinates to 4 decimal places
        coords_rounded = np.round(coords, 4).tolist()

        # Build 2D scatter plot colored by category
        df_plot = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1]
        })

        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            df_plot["category"] = session.dataframe["category_name"].values
        elif session.labels is not None:
            df_plot["category"] = session.labels
        else:
            df_plot["category"] = "default"

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=df_plot,
            x="Dim 1",
            y="Dim 2",
            hue="category",
            style="category",
            s=100,
            ax=ax,
            palette="Set1"
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        # Set required pending figure on session
        session.pending_figure = fig

        # Save coordinates array in session.artifacts
        session.artifacts[f"reduced_coords_{method}"] = coords_rounded

        result_id = session.next_result_id("reduction")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords_rounded,
            "explained_variance_ratio": explained_variance_ratio
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary
        )

        return summary

    @tool
    def binarize_labels_tool(
        column_name: str = "category_name"
    ) -> Dict[str, Any]:
        """
        One-hot encodes categorical labels into a binary matrix (0/1 indicators per category).

        Args:
            column_name: The column name in session.dataframe containing categorical labels.

        Returns:
            Dict containing sample count, category list, category count, and 2D one-hot binary matrix.
        """
        labels = None
        if session.dataframe is not None and column_name in session.dataframe.columns:
            labels = session.dataframe[column_name]
        elif session.labels is not None:
            labels = session.labels

        if labels is None:
            return {
                "status": "error",
                "error": f"No categorical labels found under '{column_name}' or session.labels."
            }

        # Produce a complete one-hot matrix with 0/1 integers for every category
        dummies_df = pd.get_dummies(labels, dtype=int)
        categories = dummies_df.columns.astype(str).tolist()
        one_hot_matrix = dummies_df.values.tolist()

        # Store artifacts without mutating session.labels or session.feature_matrix
        session.artifacts["binarized_labels"] = one_hot_matrix
        session.artifacts["binarized_categories"] = categories

        result_id = session.next_result_id("binarize")
        summary = {
            "result_id": result_id,
            "column_name": column_name,
            "n_samples": len(dummies_df),
            "categories": categories,
            "n_categories": len(categories),
            "one_hot_matrix": one_hot_matrix
        }

        session.store_result(
            "binarize_labels_tool",
            {"column_name": column_name},
            summary
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
