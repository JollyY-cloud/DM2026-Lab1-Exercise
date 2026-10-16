from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Filters features from the Document-Term Matrix based on variance across documents.

        Args:
            threshold: Minimum variance threshold required to keep a term. Terms with variance <= threshold are flagged for removal.

        Returns:
            Dict containing summary stats, kept features, and removed features.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        dtm_dense = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)

        # Calculate variance per term (column)
        variances = np.var(dtm_dense, axis=0)

        kept_terms = []
        removed_terms = []
        report_data = []

        for term, var in zip(feature_names, variances):
            var_val = float(var)
            report_data.append({"term": term, "variance": round(var_val, 4)})
            if var_val > threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        # Create full_report DataFrame sorted by variance descending for visualize_result_tool
        df_report = pd.DataFrame(report_data).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_original_features": len(feature_names),
            "n_kept_features": len(kept_terms),
            "n_removed_features": len(removed_terms),
            "kept_features": kept_terms,
            "removed_features": removed_terms
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str
    ) -> Dict[str, Any]:
        """
        Computes Pearson correlation between each term's frequency and a target class.

        Args:
            target_class: The target category name to correlate against (one-vs-rest).

        Returns:
            Dict containing summary information and top correlated terms for the target class.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        labels = None
        if session.labels is not None:
            labels = np.array(session.labels)
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        
        if labels is None:
            return {
                "status": "error",
                "error": "No class labels found in session. Ensure dataset is loaded with labels."
            }

        if target_class not in labels:
            return {
                "status": "error",
                "error": f"Target class '{target_class}' not found in dataset labels: {sorted(list(set(labels)))}"
            }

        # One-vs-rest binary target vector
        y_binary = (labels == target_class).astype(float)

        dtm_dense = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)

        report_data = []

        # Check if target vector has zero variance
        y_var = np.var(y_binary)

        for i, term in enumerate(feature_names):
            x_col = dtm_dense[:, i]
            x_var = np.var(x_col)

            if x_var == 0 or y_var == 0:
                r_val = 0.0
            else:
                r, _ = pearsonr(x_col, y_binary)
                r_val = 0.0 if np.isnan(r) else float(r)

            report_data.append({"term": term, "pearson_r": round(float(r_val), 4)})

        # Sort report DataFrame by pearson_r descending for visualize_result_tool
        df_report = pd.DataFrame(report_data).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Extract top positive and top negative terms for concise summary
        top_positive = df_report.head(5).to_dict(orient="records")
        top_negative = df_report.tail(5).sort_values(by="pearson_r", ascending=True).to_dict(orient="records")

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_features": len(feature_names),
            "top_positive_correlations": top_positive,
            "top_negative_correlations": top_negative
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str
    ) -> Dict[str, Any]:
        """
        Computes Spearman rank correlation between each term's frequency and a target class.

        Args:
            target_class: The target category name to correlate against (one-vs-rest).

        Returns:
            Dict containing summary information and top rank-correlated terms for the target class.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        labels = None
        if session.labels is not None:
            labels = np.array(session.labels)
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        
        if labels is None:
            return {
                "status": "error",
                "error": "No class labels found in session. Ensure dataset is loaded with labels."
            }

        if target_class not in labels:
            return {
                "status": "error",
                "error": f"Target class '{target_class}' not found in dataset labels: {sorted(list(set(labels)))}"
            }

        # One-vs-rest binary target vector
        y_binary = (labels == target_class).astype(float)

        dtm_dense = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)

        report_data = []

        # Check if target vector has zero variance
        y_var = np.var(y_binary)

        for i, term in enumerate(feature_names):
            x_col = dtm_dense[:, i]
            x_var = np.var(x_col)

            if x_var == 0 or y_var == 0:
                r_val = 0.0
            else:
                r, _ = spearmanr(x_col, y_binary)
                r_val = 0.0 if np.isnan(r) else float(r)

            report_data.append({"term": term, "spearman_r": round(float(r_val), 4)})

        # Sort report DataFrame by spearman_r descending for visualize_result_tool
        df_report = pd.DataFrame(report_data).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Extract top positive and top negative terms for concise summary
        top_positive = df_report.head(5).to_dict(orient="records")
        top_negative = df_report.tail(5).sort_values(by="spearman_r", ascending=True).to_dict(orient="records")

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_features": len(feature_names),
            "top_positive_correlations": top_positive,
            "top_negative_correlations": top_negative
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class},
            summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
