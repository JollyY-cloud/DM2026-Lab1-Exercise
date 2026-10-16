from typing import Dict, Any, List, Optional
import os
import pathlib
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """
        Checks for missing, null, or empty text values in session.dataframe.

        Returns:
            Dict containing total rows, count and percentage of missing text rows, and missing row indices.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "status": "error",
                "error": "Column 'text' not found in session.dataframe."
            }

        # Check for nulls, NaNs, or whitespace-only strings
        text_series = df["text"]
        is_missing = text_series.isna() | (text_series.astype(str).str.strip() == "")

        missing_indices = df.index[is_missing].tolist()
        n_missing = len(missing_indices)
        total_rows = len(df)
        missing_pct = (100.0 * n_missing / total_rows) if total_rows > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "n_missing": n_missing,
            "missing_percentage": round(float(missing_pct), 4),
            "missing_indices": missing_indices
        }

        session.store_result("check_missing_tool", {}, summary)

        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False
    ) -> Dict[str, Any]:
        """
        Checks for (and optionally drops) duplicate documents in session.dataframe.

        Args:
            drop: If True, drops ALL copies of duplicated rows using keep=False.

        Returns:
            Dict containing count of duplicate rows, duplicate indices, action taken, and remaining rows.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "status": "error",
                "error": "Column 'text' not found in session.dataframe."
            }

        # Identify all duplicated rows (keep=False marks all copies of duplicates)
        dup_mask = df.duplicated(subset=["text"], keep=False)
        duplicate_indices = df.index[dup_mask].tolist()
        n_duplicates = len(duplicate_indices)

        if drop:
            # Use keep=False to drop every copy of a duplicated row
            session.dataframe = df.drop_duplicates(subset=["text"], keep=False)
            action_taken = "dropped_all_copies"
        else:
            action_taken = "checked_only"

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "action_taken": action_taken,
            "remaining_rows": len(session.dataframe)
        }

        session.store_result(
            "check_duplicates_tool",
            {"drop": drop},
            summary
        )

        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """
        Tokenizes document text into unigrams using nltk.word_tokenize on lowercased text.

        Returns:
            Dict containing document count, total tokens, average tokens per document, and sample tokens.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "status": "error",
                "error": "Column 'text' not found in session.dataframe."
            }

        # Ensure required NLTK tokenizer resources are present
        for resource_path, download_name in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(download_name, quiet=True)

        tokenized_docs = []
        for text in df["text"]:
            if pd.isna(text) or str(text).strip() == "":
                tokenized_docs.append([])
            else:
                text_clean = str(text).lower()
                tokens = word_tokenize(text_clean)
                tokenized_docs.append(tokens)

        # Store in session.dataframe and session.artifacts
        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["unigrams"] = tokenized_docs

        total_tokens = sum(len(t) for t in tokenized_docs)
        n_docs = len(tokenized_docs)
        avg_tokens = (total_tokens / n_docs) if n_docs > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(float(avg_tokens), 4),
            "sample_unigrams": tokenized_docs[:3]
        }

        session.store_result("tokenize_tool", {}, summary)

        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Subsamples the working DataFrame (session.dataframe) without modifying it.

        Args:
            n: Number of sample rows to return.
            random_state: Seed for reproducible random sampling.

        Returns:
            Dict containing sampled indices, count, and sampled records.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        if n <= 0:
            return {
                "status": "error",
                "error": "Sample size n must be a positive integer."
            }

        df = session.dataframe
        total_rows = len(df)
        sample_n = min(n, total_rows)

        # Sample rows using pandas DataFrame.sample without modifying session.dataframe
        sampled_df = df.sample(n=sample_n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        cols_to_include = [c for c in ["text", "category_name"] if c in sampled_df.columns]
        sampled_records = sampled_df[cols_to_include].to_dict(orient="records") if cols_to_include else sampled_df.to_dict(orient="records")

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "n_requested": n,
            "n_sampled": sample_n,
            "random_state": random_state,
            "sampled_indices": sampled_indices,
            "sampled_data": sampled_records
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state},
            summary
        )

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """
        Computes summary statistics on text length and generates a boxplot by category.

        Returns:
            Dict containing overall and per-category text length summary statistics.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "status": "error",
                "error": "Column 'text' not found in session.dataframe."
            }

        # Calculate text length per document (character count)
        df["text_length"] = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        # Helper to format .describe() series into rounded dict
        def format_describe(series):
            desc = series.describe()
            return {
                "count": int(desc["count"]),
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"])
            }

        overall_stats = format_describe(df["text_length"])

        per_category_stats = {}
        has_categories = "category_name" in df.columns

        if has_categories:
            for cat, group in df.groupby("category_name"):
                per_category_stats[str(cat)] = format_describe(group["text_length"])

        # Build box plot of text_length grouped by category_name
        fig, ax = plt.subplots(figsize=(8, 5))
        if has_categories:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax, palette="Set2")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=df["text_length"], ax=ax, palette="Set2")
            ax.set_xlabel("Overall")

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("describe")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)

        return summary

    @tool
    def list_files_tool(
        subdirectory: str = "."
    ) -> Dict[str, Any]:
        """
        Lists files and subdirectories in a workspace/repository directory.

        Args:
            subdirectory: Subdirectory path relative to repository root.

        Returns:
            Dict containing lists of immediate directories and files in subdirectory.
        """
        # Resolve target directory path relative to current working directory (repo root)
        target_path = pathlib.Path.cwd() / subdirectory

        if not target_path.exists() or not target_path.is_dir():
            return {
                "status": "error",
                "error": f"Subdirectory '{subdirectory}' does not exist or is not a directory."
            }

        directories = []
        files = []

        for item in target_path.iterdir():
            # Skip hidden files/directories and cache directories
            if item.name.startswith(".") or item.name in ["__pycache__", ".pytest_cache", "venv", ".venv"]:
                continue
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result(
            "list_files_tool",
            {"subdirectory": subdirectory},
            summary
        )

        return summary

    @tool
    def inspect_data_tool(
        n: int = 5
    ) -> Dict[str, Any]:
        """
        Inspects the working DataFrame (session.dataframe), returning a head row preview and column data types.

        Args:
            n: Number of head rows to preview.

        Returns:
            Dict containing total rows, column list, data types mapping, and head row preview records.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first."
            }

        if n <= 0:
            return {
                "status": "error",
                "error": "Preview row count n must be a positive integer."
            }

        df = session.dataframe
        total_rows = len(df)
        peek_n = min(n, total_rows)

        dtypes_map = {col: str(dtype) for col, dtype in df.dtypes.items()}

        preview_records = df.head(peek_n).to_dict(orient="records")

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "n_preview": peek_n,
            "columns": df.columns.tolist(),
            "dtypes": dtypes_map,
            "preview": preview_records
        }

        session.store_result(
            "inspect_data_tool",
            {"n": n},
            summary
        )

        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        describe_data_tool,
        list_files_tool,
        inspect_data_tool
    ]
