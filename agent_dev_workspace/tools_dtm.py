from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0
    ) -> Dict[str, Any]:
        """
        Builds a Document-Term Matrix (DTM) from text documents in session.dataframe.

        Args:
            max_features: Maximum number of terms to keep in vocabulary, ordered by term frequency across the corpus.
            ngram_range: List of 2 integers [min_n, max_n] specifying lower and upper boundary of n-gram sizes.
            min_df: Minimum document frequency for a term to be included (count or fraction).
            max_df: Maximum document frequency for a term to be included (count or fraction).

        Returns:
            Dict summarizing dataset size, vocabulary size, non-zero entries, and sparsity percentage.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        # Input validation for ngram_range
        if not (isinstance(ngram_range, (list, tuple)) and len(ngram_range) == 2):
            return {
                "status": "error",
                "error": "Invalid ngram_range. Must be a list of two positive integers [min_n, max_n] where min_n <= max_n."
            }
        
        min_n, max_n = ngram_range
        if not (isinstance(min_n, int) and isinstance(max_n, int) and min_n >= 1 and max_n >= 1 and min_n <= max_n):
            return {
                "status": "error",
                "error": "Invalid ngram_range. Must be a list of two positive integers [min_n, max_n] where min_n <= max_n."
            }

        text_series = session.dataframe["text"].fillna("").astype(str)

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(min_n, max_n),
            min_df=min_df,
            max_df=max_df
        )

        dtm = vectorizer.fit_transform(text_series)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = dtm
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        session.artifacts["count_vectorizer"] = vectorizer

        n_samples, n_features = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = n_samples * n_features
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_samples,
            "n_features": n_features,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(float(sparsity_pct), 4)
        }

        session.store_result("build_dtm_tool", {
            "max_features": max_features,
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df
        }, summary)

        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """
        Aggregates term frequencies across all documents in the Document-Term Matrix.

        Returns:
            Dict containing summary stats and top term frequencies across the entire corpus.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        # Calculate column sums (total frequency for each term)
        counts = np.array(session.feature_matrix.sum(axis=0)).flatten()
        feature_names = list(session.feature_names)

        report_data = [{"term": term, "frequency": int(cnt)} for term, cnt in zip(feature_names, counts)]

        # Create full_report DataFrame sorted by frequency descending for visualize_result_tool
        df_report = pd.DataFrame(report_data).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        top_terms = df_report.head(10).to_dict(orient="records")

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "n_terms": len(feature_names),
            "total_term_occurrences": int(np.sum(counts)),
            "top_terms": top_terms
        }

        session.store_result(
            "term_frequency_tool",
            {},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """
        Generates a heatmap visualization of a slice of the Document-Term Matrix.

        Args:
            n_terms: Number of terms (columns) to include from vocabulary order.
            n_documents: Number of documents (rows) to include from dataset row order.

        Returns:
            Dict containing the matrix slice, included terms, document labels, and result metadata.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "error": "No feature matrix found in session. Call build_dtm_tool first."
            }

        total_docs, total_terms = session.feature_matrix.shape

        slice_docs = min(n_documents, total_docs)
        slice_terms = min(n_terms, total_terms)

        # Slice sub-matrix and convert to dense array
        matrix_slice_arr = session.feature_matrix[0:slice_docs, 0:slice_terms].toarray()
        included_terms = list(session.feature_names[0:slice_terms])
        doc_labels = [f"Doc {i}" for i in range(slice_docs)]

        # Build heatmap plot
        fig, ax = plt.subplots(figsize=(max(6, slice_terms * 0.8), max(4, slice_docs * 0.5)))
        sns.heatmap(
            matrix_slice_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=included_terms,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (Top {slice_docs} Docs x {slice_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": slice_docs,
            "n_terms": slice_terms,
            "included_terms": included_terms,
            "doc_labels": doc_labels,
            "matrix_slice": matrix_slice_arr.tolist()
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
