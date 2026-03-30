"""
ML Pipeline for tabular classification tasks.

Usage example:
    from ml_pipeline import MLPipeline

    pipeline = MLPipeline(df)
    pipeline.fit(
        input_cols=["col1", "col2", "col3"],
        output_col="target",
        fill_strategy={
            "col1": "median",
            "col2": "mode",
            "col3": "constant:unknown",
        },
        test_size=0.2,
        models="all",       # or list: ["random_forest", "xgboost"]
    )
    pipeline.compare_models()
    pipeline.predict(new_df)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings

# XGBoost is optional — gracefully skip if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Run: pip install xgboost")


# ---------------------------------------------------------------------------
# Available models registry
# ---------------------------------------------------------------------------

def _build_model_registry() -> dict:
    registry = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree":       DecisionTreeClassifier(random_state=42),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "svm":                 SVC(probability=True, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        registry["xgboost"] = XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", verbosity=0
        )
    return registry


# ---------------------------------------------------------------------------
# Fill strategies
# ---------------------------------------------------------------------------

FILL_STRATEGIES = {
    "mean":           "Fills with column mean (numeric only)",
    "median":         "Fills with column median (numeric only)",
    "mode":           "Fills with most frequent value",
    "drop":           "Drops rows with missing values in this column",
    "constant:VALUE": "Fills with a constant, e.g. 'constant:0' or 'constant:unknown'",
}


def _apply_fill_strategy(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    """Apply a single fill strategy to one column. Returns modified DataFrame."""
    df = df.copy()

    if strategy == "drop":
        return df.dropna(subset=[col])

    if strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())

    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())

    elif strategy == "mode":
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    elif strategy.startswith("constant:"):
        constant_value = strategy.split(":", 1)[1]
        try:
            constant_value = float(constant_value)
        except ValueError:
            pass
        df[col] = df[col].fillna(constant_value)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}' for column '{col}'.\n"
            f"Available strategies: {list(FILL_STRATEGIES.keys())}"
        )

    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MLPipeline:
    """
    A simple ML pipeline for classification on tabular data.

    Categorical columns are encoded with OneHotEncoder (one binary column
    per unique value). Numeric columns are scaled with StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset (loaded as a Pandas DataFrame).
    """

    def __init__(self, df: pd.DataFrame):
        self.df_original = df.copy()
        self.models = _build_model_registry()
        self.trained_models: dict = {}
        self.results: dict = {}
        self.target_encoder = None      # LabelEncoder for the output column
        self.preprocessor = None        # ColumnTransformer (OHE + scaler)
        self.X_test = None
        self.y_test = None
        self._input_cols = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        input_cols: list[str],
        output_col: str,
        fill_strategy: dict[str, str] | None = None,
        test_size: float = 0.2,
        models: list[str] | str = "all",
        random_state: int = 42,
    ) -> "MLPipeline":
        """
        Preprocess the data and train selected models.

        Parameters
        ----------
        input_cols : list[str]
            Column names to use as features (X).
        output_col : str
            Column name to use as the target (y).
        fill_strategy : dict, optional
            Per-column fill strategy, e.g.:
                {
                    "age":    "median",
                    "city":   "mode",
                    "score":  "mean",
                    "status": "constant:unknown",
                    "notes":  "drop",
                }
            Columns not listed are left as-is (NaNs may cause errors).
        test_size : float
            Fraction of data reserved for testing (default 0.2 = 20%).
        models : list[str] or "all"
            Which models to train. Use "all" or e.g. ["random_forest", "xgboost"].
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        self (for method chaining)
        """
        self._input_cols = input_cols
        df = self.df_original.copy()

        # 1. Apply per-column fill strategies
        if fill_strategy:
            for col, strategy in fill_strategy.items():
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")
                df = _apply_fill_strategy(df, col, strategy)

        # 2. Keep only the columns we need
        df = df[input_cols + [output_col]]

        # 3. Split into X (features) and y (target)
        X = df[input_cols]
        y = df[output_col]

        # 4. Encode target column if it is categorical (text/category)
        if y.dtype == object or str(y.dtype) == "category":
            self.target_encoder = LabelEncoder()
            y = pd.Series(
                self.target_encoder.fit_transform(y),
                name=output_col
            )
            print(f"Target classes: {list(self.target_encoder.classes_)}")

        # 5. Detect which input columns are categorical vs numeric
        categorical_cols = [
            col for col in input_cols
            if X[col].dtype == object or str(X[col].dtype) == "category"
        ]
        numeric_cols = [
            col for col in input_cols
            if col not in categorical_cols
        ]

        print(f"Numeric columns  : {len(numeric_cols)}")
        print(f"Categorical columns (OneHotEncoded): {len(categorical_cols)}")

        # 6. Build the ColumnTransformer:
        #    - OneHotEncoder for categorical columns
        #      handle_unknown="ignore" → unknown values in predict() become all zeros
        #    - StandardScaler for numeric columns
        transformers = []
        if categorical_cols:
            transformers.append((
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ))
        if numeric_cols:
            transformers.append((
                "scaler",
                StandardScaler(),
                numeric_cols,
            ))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        # 7. Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.X_test = X_test
        self.y_test  = y_test
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # 8. Fit preprocessor on train, transform both sets
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed  = self.preprocessor.transform(X_test)

        print(f"Feature matrix shape after encoding: {X_train_processed.shape}")

        # 9. Train each selected model
        selected_models = self._select_models(models)
        print("\nTraining models...")
        for name, model in selected_models.items():
            print(f"  → {name}...", end=" ")
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            acc = accuracy_score(y_test, y_pred)
            self.trained_models[name] = model
            self.results[name] = {
                "accuracy": acc,
                "y_pred":   y_pred,
                "report":   classification_report(y_test, y_pred, output_dict=True),
            }
            print(f"accuracy = {acc:.4f}")

        self._fitted = True
        return self

    def compare_models(self) -> pd.DataFrame:
        """
        Print and return a comparison table of all trained models.

        Returns
        -------
        pd.DataFrame with accuracy, precision, recall, f1 per model.
        """
        self._check_fitted()

        rows = []
        for name, result in self.results.items():
            report = result["report"]
            rows.append({
                "model":     name,
                "accuracy":  round(result["accuracy"], 4),
                "precision": round(report["weighted avg"]["precision"], 4),
                "recall":    round(report["weighted avg"]["recall"], 4),
                "f1_score":  round(report["weighted avg"]["f1-score"], 4),
            })

        comparison = (
            pd.DataFrame(rows)
            .sort_values("accuracy", ascending=False)
            .reset_index(drop=True)
        )

        print("\n=== Model Comparison ===")
        print(comparison.to_string(index=False))
        return comparison

    def full_report(self, model_name: str) -> None:
        """
        Print a full classification report and confusion matrix for one model.

        Parameters
        ----------
        model_name : str
            One of the trained model names (e.g. "random_forest").
        """
        self._check_fitted()
        if model_name not in self.trained_models:
            raise ValueError(
                f"Model '{model_name}' was not trained. "
                f"Available: {list(self.trained_models.keys())}"
            )

        y_pred = self.results[model_name]["y_pred"]
        print(f"\n=== Full Report: {model_name} ===")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def predict(self, new_df: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Predict on new data using a trained model.

        Parameters
        ----------
        new_df : pd.DataFrame
            New data with the same input columns used during training.
        model_name : str, optional
            Which model to use. Defaults to the best model by accuracy.

        Returns
        -------
        np.ndarray of predicted labels (original class names if target was categorical).
        """
        self._check_fitted()

        if model_name is None:
            model_name = max(self.results, key=lambda m: self.results[m]["accuracy"])
            print(f"Using best model: {model_name}")

        # Reuse the same preprocessor fitted during .fit()
        X_processed = self.preprocessor.transform(new_df[self._input_cols])
        predictions = self.trained_models[model_name].predict(X_processed)

        # Decode numeric predictions back to original class names
        if self.target_encoder is not None:
            predictions = self.target_encoder.inverse_transform(predictions)

        return predictions

    @staticmethod
    def available_models() -> list[str]:
        """Return list of model names that can be used."""
        return list(_build_model_registry().keys())

    @staticmethod
    def available_fill_strategies() -> dict:
        """Return dict of available fill strategies with descriptions."""
        return FILL_STRATEGIES

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_models(self, models: list[str] | str) -> dict:
        if models == "all":
            return self.models
        unknown = set(models) - set(self.models.keys())
        if unknown:
            raise ValueError(
                f"Unknown model(s): {unknown}. "
                f"Available: {list(self.models.keys())}"
            )
        return {name: self.models[name] for name in models}

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before using this method.")
