from typing import Dict, Any, List, Optional, Union
import os
import tempfile
import numpy as np
import pandas as pd
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

        dtm_dense = dtm_local.toarray()

        # Step 2: Apply chosen filtering_method
        if filtering_method == "variance":
            scores = np.var(dtm_dense, axis=0)
            low_p, high_p = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= low_p) & (scores <= high_p)
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)
            low_p, high_p = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= low_p) & (scores <= high_p)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(dtm_dense).toarray()
            scores = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(scores, 20)
            keep_mask = scores >= p20
        else:
            return {
                "status": "error",
                "error": f"Invalid filtering_method '{filtering_method}'. Choose 'variance', 'tfidf', or 'term_frequency'."
            }

        kept_features = feature_names[keep_mask]
        kept_dtm = dtm_dense[:, keep_mask]

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

        # Step 4: Write transactional database file for PAMI
        # One transaction per document with count >= 1
        transactions = []
        for row in kept_dtm:
            present_terms = [kept_features[j] for j in range(len(kept_features)) if row[j] >= 1]
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

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp_file:
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
                os.remove(tmp_filename)
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
        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

        # Store pattern list in session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "patterns": patterns_list
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary
        )

        return summary

    return [mine_patterns_tool]
