"""Prediction explanation utilities using SHAP.

Provides a unified `PredictionExplainer` that can explain predictions
from classical sklearn pipelines (TF-IDF + NB) and transformer-based
classifiers (`TransformerClassifier`). It returns token-level importances
and a simple HTML-highlighted visualization.
"""
from typing import List, Tuple, Dict, Any
import html

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


class PredictionExplainer:
    """Explain predictions from different model backends using SHAP.

    Usage:
        explainer = PredictionExplainer()
        toks, contribs, html = explainer.explain_text(model, text, model_type='classical')
    """

    def __init__(self, masker: Any = None):
        if not SHAP_AVAILABLE:
            raise ImportError("shap is required for explanations. Install with: pip install shap")
        # Use the provided masker or the text masker from SHAP
        self.masker = masker or shap.maskers.Text()

    def _make_predict_fn(self, model: Any, model_type: str):
        """Return a callable that maps List[str] -> np.ndarray of positive-class probabilities."""
        if model_type == 'classical':
            # Expect sklearn Pipeline-like with predict_proba
            if not hasattr(model, 'pipeline') and not hasattr(model, 'predict_proba'):
                raise ValueError("Classical model must have a `pipeline` or `predict_proba` method.")

            def predict_fn(texts: List[str]):
                # If the model exposes pipeline, use pipeline.predict_proba
                if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'predict_proba'):
                    return model.pipeline.predict_proba(texts)[:, 1]
                return model.predict_proba(texts)

            return predict_fn

        elif model_type == 'transformer':
            # Expect TransformerClassifier with predict_proba
            if not hasattr(model, 'predict_proba'):
                raise ValueError("Transformer model must implement `predict_proba(texts)`.")

            def predict_fn(texts: List[str]):
                return model.predict_proba(texts)

            return predict_fn

        else:
            raise ValueError("Unsupported model_type. Use 'classical' or 'transformer'.")

    def explain_text(self, model: Any, text: str, model_type: str = 'classical', top_k: int = 10) -> Dict[str, Any]:
        """Explain a single text prediction.

        Returns a dict with:
          - `tokens`: list of tokens
          - `values`: contribution values (float, positive favors 'disaster')
          - `top_tokens`: list of (token, value) sorted by abs(value)
          - `html`: highlighted HTML string
        """
        predict_fn = self._make_predict_fn(model, model_type)

        # Create SHAP explainer for text
        explainer = shap.Explainer(predict_fn, self.masker)

        # Compute explanation
        shap_values = explainer([text])

        # shap_values.data may be tokens or raw text depending on masker
        try:
            tokens = list(shap_values.data[0])
        except Exception:
            # Fallback: split on whitespace
            tokens = text.split()

        # shap_values.values shape: (1, n_tokens) or (1, n_tokens, 1)
        vals = shap_values.values
        if hasattr(vals, 'ndim') and vals.ndim == 3:
            # (1, n_tokens, 1) -> squeeze last dim
            values = [float(v[0]) for v in vals[0]]
        else:
            values = [float(v) for v in vals[0]]

        # Pair tokens with contributions
        token_contribs = list(zip(tokens, values))

        # Determine top contributors
        top = sorted(token_contribs, key=lambda x: abs(x[1]), reverse=True)[:top_k]

        html_viz = self._tokens_to_highlighted_html(token_contribs)

        return {
            'tokens': tokens,
            'values': values,
            'top_tokens': top,
            'html': html_viz
        }

    def _tokens_to_highlighted_html(self, token_contribs: List[Tuple[str, float]]) -> str:
        """Create a simple HTML visualization that highlights tokens.

        Positive contributions (push toward 'disaster') are red shades; negative contributions
        (toward 'non-disaster') are blue shades. Intensity scaled by relative magnitude.
        """
        if not token_contribs:
            return html.escape("")

        magnitudes = [abs(v) for _, v in token_contribs]
        max_mag = max(magnitudes) if max(magnitudes) > 0 else 1.0

        parts: List[str] = []
        for token, val in token_contribs:
            # scale magnitude to 0-255
            intensity = int(min(255, (abs(val) / max_mag) * 200 + 30))
            safe_token = html.escape(token)
            if val > 0:
                # red background
                color = f"rgba(255,0,0,{intensity/300:.2f})"
            elif val < 0:
                # blue background
                color = f"rgba(0,0,255,{intensity/300:.2f})"
            else:
                color = "transparent"

            # Add data-value and class for improved rendering and tooltips
            cls = 'pos' if val > 0 else ('neg' if val < 0 else 'neu')
            span = (
                f"<span class=\"shap-token {cls}\" data-value=\"{val:.4f}\" "
                f"title=\"{val:.4f}\" style=\"background:{color};padding:2px;border-radius:3px;margin:1px\">{safe_token}</span>"
            )
            parts.append(span)

        return "".join(parts)


def explain_prediction(model: Any, text: str, model_type: str = 'classical', top_k: int = 10) -> Dict[str, Any]:
    """Convenience function to explain a prediction with default explainer."""
    expl = PredictionExplainer()
    return expl.explain_text(model, text, model_type=model_type, top_k=top_k)
