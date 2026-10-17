from typing import Dict, Any, List, Optional, Union
import os
import tempfile
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports
try:
    from PAMI.frequentPattern.basic import FPGrowth
    from PAMI.frequentPattern.topk import FAE as TopK
    from PAMI.frequentPattern.maximal import MaxFPGrowth
except ImportError:
    FPGrowth = None
    TopK = None
    MaxFPGrowth = None


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Mines frequent itemsets/patterns from documents of a specific category using PAMI.

        Args:
            category_name: Category name to filter and mine.
            filtering_method: Method to filter local vocabulary ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (count) for fpgrowth/maxfpgrowth.
            k: Number of top patterns for topk algorithm.

        Returns:
            Dict containing mined patterns, support values, and filtering metadata.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else "label"
        if cat_col not in df.columns:
            return {
                "status": "error",
                "error": "Category column not found in session.dataframe."
            }

        cat_df = df[df[cat_col] == category_name]
        if len(cat_df) == 0:
            return {
                "status": "error",
                "error": f"No documents found for category '{category_name}'."
            }

        text_series = cat_df["text"].fillna("").astype(str)

        # Step 1: Category-local DTM with English stopword removal
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_local = vectorizer.fit_transform(text_series)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Handles empty vocabulary (e.g. if all words in category are stopwords)
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        if len(feature_names) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Ensure CSR format for efficient sparse operations
        X_sparse = sp.csr_matrix(dtm_local)

        # Step 2: Apply chosen filtering_method using sparse operations
        if filtering_method == "variance":
            # Variance = E[X^2] - (E[X])^2 computed directly on sparse CSR columns
            mean_x = np.array(X_sparse.mean(axis=0)).ravel()
            mean_x2 = np.array(X_sparse.power(2).mean(axis=0)).ravel()
            scores = mean_x2 - (mean_x ** 2)
            low_p, high_p = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= low_p) & (scores <= high_p)

        elif filtering_method == "term_frequency":
            # Sum of term counts across documents directly on sparse matrix
            scores = np.array(X_sparse.sum(axis=0)).ravel()
            low_p, high_p = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= low_p) & (scores <= high_p)

        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(X_sparse)
            scores = np.array(tfidf_mat.mean(axis=0)).ravel()
            p20 = np.percentile(scores, 20)
            keep_mask = scores >= p20

        else:
            return {
                "status": "error",
                "error": f"Invalid filtering_method '{filtering_method}'. Choose 'variance', 'tfidf', or 'term_frequency'."
            }

        kept_features = feature_names[keep_mask]
        kept_dtm = X_sparse[:, keep_mask]

        # Step 3: Handle zero-surviving terms
        if len(kept_features) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm
            }, summary)
            return summary

        # Step 4: Write transactional database file for PAMI efficiently using CSR pointers
        # One transaction per document with count >= 1
        indptr = kept_dtm.indptr
        indices = kept_dtm.indices
        data = kept_dtm.data

        transactions = []
        for i in range(kept_dtm.shape[0]):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            row_indices = indices[row_start:row_end]
            row_data = data[row_start:row_end]

            # Collect terms with count >= 1 in this row
            present_terms = [kept_features[col_idx] for col_idx, val in zip(row_indices, row_data) if val >= 1]
            if present_terms:
                transactions.append("\t".join(present_terms))

        if not transactions:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no non-empty transactions found"
            }
            session.store_result("mine_patterns_tool", {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm
            }, summary)
            return summary

        with tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            delete=False,
            suffix=".txt"
        ) as tmp_file:
            tmp_filename = tmp_file.name
            for t in transactions:
                tmp_file.write(t + "\n")

        patterns_list = []
        try:
            # Step 5: Execute PAMI mining algorithm
            sep = "\t"
            if algorithm.lower() == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=tmp_filename, minSup=min_sup, sep=sep)
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
            elif algorithm.lower() == "topk":
                miner = TopK.FAE(iFile=tmp_filename, k=k, sep=sep)
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
            elif algorithm.lower() == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=tmp_filename, minSup=min_sup, sep=sep)
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
            else:
                return {
                    "status": "error",
                    "error": f"Invalid algorithm '{algorithm}'. Choose 'fpgrowth', 'topk', or 'maxfpgrowth'."
                }

            if df_res is not None and len(df_res) > 0:
                pat_col = [c for c in df_res.columns if c.lower() == "patterns"][0]
                sup_col = [c for c in df_res.columns if c.lower() == "support"][0]

                for _, r in df_res.iterrows():
                    pat_str = str(r[pat_col])
                    items = pat_str.strip().split()
                    sup_val = int(r[sup_col])
                    patterns_list.append({
                        "pattern": items,
                        "support": sup_val
                    })

        except Exception as e:
            return {
                "status": "error",
                "error": f"PAMI pattern mining execution failed: {str(e)}"
            }

        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

        # Store complete original pattern list in session.artifacts and full_report
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("patterns")
        n_patterns = len(patterns_list)

        # For summary returned to LLM, limit to top 20 patterns sorted by descending support
        if n_patterns <= 20:
            patterns_summary = patterns_list
            note = None
        else:
            sorted_patterns = sorted(patterns_list, key=lambda x: x["support"], reverse=True)
            patterns_summary = sorted_patterns[:20]
            note = f"Returned top 20 of {n_patterns} total patterns by support. Full pattern list saved in session.artifacts['{artifact_key}']"

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": n_patterns,
            "patterns": patterns_summary
        }
        if note:
            summary["note"] = note

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary,
            full_report=patterns_list
        )

        return summary

    return [mine_patterns_tool]
