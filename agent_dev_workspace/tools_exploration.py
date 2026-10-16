from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> Dict[str, Any]:
        """
        Computes cosine similarity between two documents in the Document-Term Matrix.

        Args:
            doc_index_1: Zero-based row index of the first document.
            doc_index_2: Zero-based row index of the second document.

        Returns:
            Dict containing document indices and computed cosine similarity score.
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        n_docs = session.feature_matrix.shape[0]

        if not (0 <= doc_index_1 < n_docs and 0 <= doc_index_2 < n_docs):
            return {
                "status": "error",
                "error": f"Invalid document index. Must be between 0 and {n_docs - 1}."
            }

        vec1 = session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2]

        # Calculate norm to safely handle empty-text / zero-vector edge cases
        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim_val = 0.0
        else:
            sim_mat = cosine_similarity(vec1, vec2)
            sim_val = float(sim_mat[0, 0])
            if np.isnan(sim_val):
                sim_val = 0.0

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": round(sim_val, 4)
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """
        Computes Pearson correlation heatmap between top-20 terms ordered by variance.

        Returns:
            Dict containing top terms by variance, term count, and 2D correlation matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        dtm_dense = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        # Step 1: Calculate term variance across all documents
        variances = np.var(dtm_dense, axis=0)

        # Step 2: Select top 20 terms by variance, sorted descending
        top_k = min(20, len(feature_names))
        top_indices = np.argsort(variances)[::-1][:top_k]

        top_terms = feature_names[top_indices].tolist()
        sub_matrix = dtm_dense[:, top_indices]

        # Step 3: Compute Pearson correlation matrix across top terms
        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)

        # Handle NaNs if zero-variance terms present
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Step 4: Build Seaborn heatmap
        fig, ax = plt.subplots(figsize=(max(6, top_k * 0.8), max(5, top_k * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {top_k} Terms by Variance)")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        # Round matrix elements to 4 decimal places for serialization and testing
        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "n_features": top_k,
            "top_terms_by_variance": top_terms,
            "correlation_matrix": corr_matrix_rounded
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
