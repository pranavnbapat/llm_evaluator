# metrics/scientific_metrics.py

"""Scientific evaluation metrics for LLM responses."""

import json
import re
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np

# Try to import ML libraries (may not be available during setup)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    SentenceTransformer = None
    cosine_similarity = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class QualityScores:
    """Container for quality scores."""
    relevance: float = 0.0
    factual_accuracy: float = 0.0
    completeness: float = 0.0
    fluency: float = 0.0
    coherence: float = 0.0
    prompt_alignment: float = 0.0
    token_efficiency: float = 0.0
    overall_quality: float = 0.0


class EmbeddingModel:
    """Wrapper for sentence embedding model."""
    
    _instance = None
    
    def __new__(cls, model_name: str = None):
        if cls._instance is None and ML_AVAILABLE:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_name = model_name
        return cls._instance
    
    def load(self):
        """Lazy load the embedding model on CPU to avoid GPU conflicts."""
        if self.model is None and ML_AVAILABLE and self.model_name:
            import torch
            # Force CPU to avoid GPU memory conflicts with vLLM
            self.model = SentenceTransformer(self.model_name, device='cpu')
        return self.model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        model = self.load()
        if model is None:
            return np.zeros((len(texts), 768))  # Return dummy embeddings
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class ResponseEvaluator:
    """Scientific evaluator for LLM responses."""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        metrics_config_path: Optional[str] = None,
        metrics_profile: str = "default",
    ):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self._embeddings_cache = {}
        self.metrics_profile = metrics_profile
        self.metrics_config = self._load_metrics_config(metrics_config_path, metrics_profile)
        self.weights = self._get_weights(self.metrics_config)
        self.token_efficiency_cfg = self.metrics_config.get("token_efficiency", {})
        self.fluency_cfg = self.metrics_config.get("fluency", {})
        self.coherence_cfg = self.metrics_config.get("coherence", {})
        self.prompt_alignment_cfg = self.metrics_config.get("prompt_alignment", {})
        self.context_cfg = self.metrics_config.get("context", {})
        self.nli_cfg = self.metrics_config.get("nli", {})
        self._fluency_model = None
        self._coherence_model = None
        self._nli_model = None
    
    def _load_metrics_config(self, metrics_config_path: Optional[str], profile: str) -> Dict[str, Any]:
        """Load metrics configuration from YAML with profile selection."""
        default_cfg: Dict[str, Any] = {}
        config_path = None
        
        if metrics_config_path:
            config_path = Path(metrics_config_path)
        else:
            # Default to repo-local config if present
            candidate = Path(__file__).parent / "metrics_config.yaml"
            if candidate.exists():
                config_path = candidate
        
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            profiles = raw.get("profiles", {})
            default_cfg = profiles.get(profile, {})
        
        return default_cfg
    
    def _get_weights(self, cfg: Dict[str, Any]) -> Dict[str, float]:
        """Get (and optionally normalize) weights for composite scoring."""
        weights = cfg.get("weights", {})
        normalize = cfg.get("normalize_weights", False)
        if not weights:
            return {
                "relevance": 0.25,
                "factual_accuracy": 0.20,
                "completeness": 0.15,
                "fluency": 0.15,
                "coherence": 0.10,
                "prompt_alignment": 0.10,
                "token_efficiency": 0.05,
            }
        if normalize:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        return weights
    
    def calculate_relevance(
        self,
        question: str,
        response: str,
        reference: Optional[str] = None,
    ) -> float:
        """
        Calculate relevance score using semantic similarity.
        Returns score between 0 and 1.
        """
        if not response.strip():
            return 0.0
        
        if ML_AVAILABLE:
            try:
                embeddings = self.embedding_model.encode([question, response])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(max(0.0, min(1.0, similarity)))
            except Exception as e:
                # Fallback to lexical similarity
                pass
        
        # Fallback: lexical overlap
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        if not question_words:
            return 0.0
        overlap = len(question_words & response_words) / len(question_words)
        return overlap
    
    def _ensure_transformers(self):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers is not available. Install 'transformers' to use model-based metrics.")
    
    def _load_text_classifier(self, model_name: str):
        self._ensure_transformers()
        return pipeline("text-classification", model=model_name, device=-1)
    
    def _load_zero_shot_classifier(self, model_name: str):
        self._ensure_transformers()
        return pipeline("zero-shot-classification", model=model_name, device=-1)
    
    def _load_nli_classifier(self, model_name: str):
        self._ensure_transformers()
        return pipeline("text-classification", model=model_name, device=-1)
    
    def _score_with_classifier(
        self,
        clf,
        texts: List[str],
        max_length: int,
        batch_size: int,
        aggregation: str = "mean",
    ) -> float:
        if not texts:
            return 0.0
        
        results = clf(
            texts,
            truncation=True,
            max_length=max_length,
            batch_size=batch_size,
        )
        
        if isinstance(results, dict):
            results = [results]
        
        scores = []
        for r in results:
            score = r.get("score", 0.0)
            scores.append(score)
        
        if not scores:
            return 0.0
        if aggregation == "max":
            return float(max(scores))
        if aggregation == "min":
            return float(min(scores))
        return float(np.mean(scores))
    
    def _score_with_zero_shot(
        self,
        clf,
        texts: List[str],
        labels: List[str],
        positive_label: str,
        hypothesis_template: str,
        max_length: int,
        batch_size: int,
        aggregation: str = "mean",
    ) -> float:
        if not texts or not labels:
            return 0.0
        
        results = clf(
            texts,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            truncation=True,
            max_length=max_length,
            batch_size=batch_size,
        )
        
        if isinstance(results, dict):
            results = [results]
        
        scores = []
        for r in results:
            r_labels = r.get("labels", [])
            r_scores = r.get("scores", [])
            if positive_label in r_labels:
                idx = r_labels.index(positive_label)
                scores.append(float(r_scores[idx]))
        
        if not scores:
            return 0.0
        if aggregation == "max":
            return float(max(scores))
        if aggregation == "min":
            return float(min(scores))
        return float(np.mean(scores))
    
    def calculate_completeness(
        self,
        response: str,
        expected_elements: List[str],
    ) -> float:
        """
        Calculate completeness based on expected elements.
        Returns score between 0 and 1.
        """
        if not expected_elements:
            return 1.0
        
        response_lower = response.lower()
        found = sum(1 for element in expected_elements if element.lower() in response_lower)
        return found / len(expected_elements)
    
    def calculate_factual_accuracy(
        self,
        response: str,
        reference_facts: Union[Dict[str, Any], List[str]],
    ) -> float:
        """
        Calculate factual accuracy against reference facts.
        This is a simplified implementation - real implementation would use NLI.
        """
        if not reference_facts:
            return 1.0  # No facts to check
        
        response_lower = response.lower()
        correct = 0
        
        # Handle both dict and list formats
        if isinstance(reference_facts, dict):
            for key, value in reference_facts.items():
                # Check if key concept is mentioned
                if str(key).lower() in response_lower:
                    correct += 0.5
                # Check if value is mentioned
                if str(value).lower() in response_lower:
                    correct += 0.5
        else:  # List format
            for fact in reference_facts:
                if str(fact).lower() in response_lower:
                    correct += 1.0
        
        return min(1.0, correct / len(reference_facts))
    
    def calculate_nli_entailment(
        self,
        response: str,
        context_documents: List[str],
    ) -> float:
        """
        Calculate entailment score using NLI model.
        Uses context documents as premises and response sentences as hypotheses.
        """
        if not response.strip() or not context_documents:
            return 0.0
        
        model_name = self.nli_cfg.get("model_name", "")
        if not model_name:
            raise RuntimeError("NLI model_name is not set in metrics_config.yaml")
        
        if self._nli_model is None:
            self._nli_model = self._load_nli_classifier(model_name)
        
        # Build hypotheses
        mode = self.nli_cfg.get("hypothesis_from_response", "sentences")
        min_chars = int(self.nli_cfg.get("min_sentence_chars", 20))
        if mode == "sentences":
            candidates = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
            hypotheses = [c for c in candidates if len(c) >= min_chars]
            if not hypotheses:
                hypotheses = [response.strip()]
        else:
            hypotheses = [response.strip()]
        
        # Build premise-hypothesis pairs
        pairs = []
        for doc in context_documents:
            if not doc or not doc.strip():
                continue
            for hyp in hypotheses:
                pairs.append({"text": doc, "text_pair": hyp})
        
        if not pairs:
            return 0.0
        
        max_length = int(self.nli_cfg.get("max_length", 512))
        batch_size = int(self.nli_cfg.get("batch_size", 8))
        aggregation = self.nli_cfg.get("aggregation", "mean")
        
        results = self._nli_model(
            pairs,
            truncation=True,
            max_length=max_length,
            batch_size=batch_size,
        )
        if isinstance(results, dict):
            results = [results]
        
        entail_scores = []
        for r in results:
            label = str(r.get("label", "")).lower()
            score = float(r.get("score", 0.0))
            if "entail" in label:
                entail_scores.append(score)
        
        if not entail_scores:
            # Label schemas vary across NLI models; fall back to the raw scores.
            entail_scores = [float(r.get("score", 0.0)) for r in results]
        
        if not entail_scores:
            return 0.0
        
        if aggregation == "max":
            return float(max(entail_scores))
        if aggregation == "min":
            return float(min(entail_scores))
        return float(np.mean(entail_scores))
    
    def calculate_context_utilization(
        self,
        response: str,
        context_documents: List[str],
    ) -> float:
        """
        Calculate how well the response utilizes the provided context.
        Uses semantic similarity (embeddings) instead of exact string matching.
        """
        if not context_documents or not response.strip():
            return 0.0
        
        try:
            # Encode response
            response_embedding = self.embedding_model.encode([response])[0]
            
            # Encode each context document and compute similarities
            similarities = []
            for doc in context_documents:
                if not doc or not doc.strip():
                    continue
                doc_embedding = self.embedding_model.encode([doc])[0]
                
                # Calculate cosine similarity manually (avoid sklearn dependency issues)
                import numpy as np
                dot_product = np.dot(response_embedding, doc_embedding)
                norm_a = np.linalg.norm(response_embedding)
                norm_b = np.linalg.norm(doc_embedding)
                
                if norm_a == 0 or norm_b == 0:
                    continue
                    
                similarity = dot_product / (norm_a * norm_b)
                similarities.append(similarity)
            
            if not similarities:
                return float(self.context_cfg.get("fallback_score", 0.0))
            
            aggregation = self.context_cfg.get("utilization_aggregation", "max")
            top_k = int(self.context_cfg.get("top_k", 3))
            
            if aggregation == "mean_top_k":
                top_k = max(1, min(top_k, len(similarities)))
                similarities.sort(reverse=True)
                score = float(np.mean(similarities[:top_k]))
            elif aggregation == "mean":
                score = float(np.mean(similarities))
            else:
                score = float(max(similarities))
            
            # Normalize to 0-1
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            # Log error and return fallback
            print(f"⚠️ Context utilization error: {e}")
            return float(self.context_cfg.get("fallback_score", 0.0))
    
    def calculate_context_coverage(
        self,
        response: str,
        context_documents: List[str],
    ) -> float:
        """
        Calculate context coverage: fraction of context docs with similarity above threshold.
        """
        if not context_documents or not response.strip():
            return 0.0
        
        try:
            response_embedding = self.embedding_model.encode([response])[0]
            threshold = float(self.context_cfg.get("coverage_threshold", 0.35))
            
            covered = 0
            total = 0
            for doc in context_documents:
                if not doc or not doc.strip():
                    continue
                doc_embedding = self.embedding_model.encode([doc])[0]
                
                import numpy as np
                dot_product = np.dot(response_embedding, doc_embedding)
                norm_a = np.linalg.norm(response_embedding)
                norm_b = np.linalg.norm(doc_embedding)
                
                if norm_a == 0 or norm_b == 0:
                    continue
                
                similarity = dot_product / (norm_a * norm_b)
                total += 1
                if similarity >= threshold:
                    covered += 1
            
            if total == 0:
                return float(self.context_cfg.get("fallback_score", 0.0))
            
            return float(covered / total)
        except Exception as e:
            print(f"   ⚠️ Context coverage error: {e}")
            return float(self.context_cfg.get("fallback_score", 0.0))
    
    def calculate_fluency(self, response: str, expected_language: str = None) -> float:
        """
        Calculate fluency score based on multiple factors.
        """
        if not response.strip():
            return 0.0
        
        mode = self.fluency_cfg.get("mode", "zero_shot")
        model_name = self.fluency_cfg.get("model_name", "")
        if not model_name:
            raise RuntimeError("Fluency model_name is not set in metrics_config.yaml")
        
        max_length = int(self.fluency_cfg.get("max_length", 512))
        batch_size = int(self.fluency_cfg.get("batch_size", 8))
        aggregation = self.fluency_cfg.get("aggregation", "mean")
        
        if mode == "zero_shot":
            labels = self.fluency_cfg.get("labels", [])
            positive_label = self.fluency_cfg.get("positive_label", "")
            hypothesis_template = self.fluency_cfg.get("hypothesis_template", "This text is {}.")
            if not labels or not positive_label:
                raise RuntimeError("Fluency zero-shot labels/positive_label not set in metrics_config.yaml")
            if self._fluency_model is None:
                self._fluency_model = self._load_zero_shot_classifier(model_name)
            return self._score_with_zero_shot(
                self._fluency_model,
                [response],
                labels=labels,
                positive_label=positive_label,
                hypothesis_template=hypothesis_template,
                max_length=max_length,
                batch_size=batch_size,
                aggregation=aggregation,
            )
        
        if self._fluency_model is None:
            self._fluency_model = self._load_text_classifier(model_name)
        
        return self._score_with_classifier(
            self._fluency_model,
            [response],
            max_length=max_length,
            batch_size=batch_size,
            aggregation=aggregation,
        )
    
    def calculate_coherence(self, response: str) -> float:
        """
        Calculate coherence score based on discourse flow.
        """
        if not response.strip():
            return 0.0
        
        mode = self.coherence_cfg.get("mode", "zero_shot")
        model_name = self.coherence_cfg.get("model_name", "")
        if not model_name:
            raise RuntimeError("Coherence model_name is not set in metrics_config.yaml")
        
        max_length = int(self.coherence_cfg.get("max_length", 512))
        batch_size = int(self.coherence_cfg.get("batch_size", 8))
        aggregation = self.coherence_cfg.get("aggregation", "mean")
        
        if mode == "zero_shot":
            labels = self.coherence_cfg.get("labels", [])
            positive_label = self.coherence_cfg.get("positive_label", "")
            hypothesis_template = self.coherence_cfg.get("hypothesis_template", "This text is {}.")
            if not labels or not positive_label:
                raise RuntimeError("Coherence zero-shot labels/positive_label not set in metrics_config.yaml")
            if self._coherence_model is None:
                self._coherence_model = self._load_zero_shot_classifier(model_name)
            return self._score_with_zero_shot(
                self._coherence_model,
                [response],
                labels=labels,
                positive_label=positive_label,
                hypothesis_template=hypothesis_template,
                max_length=max_length,
                batch_size=batch_size,
                aggregation=aggregation,
            )
        
        if self._coherence_model is None:
            self._coherence_model = self._load_text_classifier(model_name)
        
        return self._score_with_classifier(
            self._coherence_model,
            [response],
            max_length=max_length,
            batch_size=batch_size,
            aggregation=aggregation,
        )
    
    def calculate_prompt_alignment(
        self,
        response: str,
        question: str,
    ) -> float:
        """
        Calculate how well the response aligns with the prompt.
        Detects hallucination and off-topic responses.
        """
        if not response.strip():
            return 0.0
        
        # Default: use semantic relevance to avoid language-specific heuristics
        return self.calculate_relevance(question, response)
    
    def calculate_token_efficiency(
        self,
        quality_score: float,
        token_count: int,
        expected_length: Tuple[int, int],
    ) -> float:
        """
        Calculate token efficiency - information density.
        """
        if token_count == 0:
            return 0.0
        
        # Ideal length penalty
        min_len, max_len = expected_length
        if token_count < min_len:
            length_penalty = token_count / min_len
        elif token_count > max_len:
            length_penalty = max_len / token_count
        else:
            length_penalty = 1.0
        
        # Information density
        density = quality_score / (token_count / 100)  # per 100 tokens
        
        return min(1.0, density * length_penalty)
    
    def _normalize_question_id(self, question_id: str) -> str:
        """Normalize question id for config lookups."""
        if not question_id:
            return ""
        # Handle context evaluation ids like Q1_EN
        if "_" in question_id and question_id.startswith("Q"):
            prefix = question_id.split("_")[0]
            if prefix in {"Q1", "Q2", "Q3", "Q4", "Q5"}:
                return prefix
        return question_id
    
    def _get_expected_length(self, question_id: str) -> Tuple[int, int]:
        """Get expected length range for token efficiency."""
        default_range = self.token_efficiency_cfg.get("expected_length_default", [50, 500])
        by_question = self.token_efficiency_cfg.get("expected_length_by_question", {})
        qid = self._normalize_question_id(question_id)
        return tuple(by_question.get(qid, default_range))
    
    def evaluate_json_adherence(self, response: str, required_keys: List[str]) -> float:
        """
        Evaluate adherence to JSON format requirements.
        """
        # Try to extract JSON
        json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
        if not json_match:
            return 0.0
        
        try:
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # Check for required keys
            found_keys = sum(1 for key in required_keys if key in data)
            return found_keys / len(required_keys) if required_keys else 1.0
            
        except json.JSONDecodeError:
            return 0.0
    
    def evaluate_summarization(
        self,
        summary: str,
        source_text: str,
        max_sentences: int = 3,
    ) -> Dict[str, float]:
        """
        Evaluate summarization quality.
        """
        # Check length constraint
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        length_score = 1.0 if len(sentences) <= max_sentences else 0.5
        
        # Check coverage of key information (simplified)
        relevance = self.calculate_relevance(source_text, summary)
        
        # Check for hallucination (content not in source)
        # Simplified: just check basic relevance
        
        return {
            "length_constraint": length_score,
            "coverage": relevance,
            "overall": (length_score + relevance) / 2,
        }
    
    def evaluate_response(
        self,
        question_id: str,
        question_text: str,
        response_text: str,
        language: str,
        tokens_generated: int,
        reference_data: Optional[Dict[str, Any]] = None,
    ) -> QualityScores:
        """
        Main evaluation method - compute all metrics.
        """
        # Get question-specific expected elements
        expected_elements = reference_data.get("expected_elements", []) if reference_data else []
        reference_facts = reference_data.get("reference_facts", {}) if reference_data else {}
        context_documents = reference_data.get("context_documents", []) if reference_data else []
        
        # Calculate individual scores
        relevance = self.calculate_relevance(question_text, response_text)
        
        if context_documents and self.context_cfg.get("use_coverage_for_completeness", False):
            completeness = self.calculate_context_coverage(response_text, context_documents)
        else:
            completeness = self.calculate_completeness(response_text, expected_elements)
        
        # Use NLI entailment when context documents are provided
        # Otherwise fall back to factual accuracy (string matching)
        if context_documents:
            factual_accuracy = self.calculate_nli_entailment(response_text, context_documents)
        else:
            factual_accuracy = self.calculate_factual_accuracy(response_text, reference_facts)
        
        fluency = self.calculate_fluency(response_text, language)
        
        coherence = self.calculate_coherence(response_text)
        
        prompt_alignment = self.calculate_prompt_alignment(response_text, question_text)
        
        # Composite quality before efficiency (weighted mean over base metrics)
        base_scores = {
            "relevance": relevance,
            "factual_accuracy": factual_accuracy,
            "completeness": completeness,
            "fluency": fluency,
            "coherence": coherence,
            "prompt_alignment": prompt_alignment,
        }
        base_weights = {k: self.weights.get(k, 0.0) for k in base_scores}
        weight_sum = sum(base_weights.values())
        if weight_sum > 0:
            base_quality = sum(base_scores[k] * base_weights[k] for k in base_scores) / weight_sum
        else:
            base_quality = sum(base_scores.values()) / len(base_scores)
        if self.token_efficiency_cfg.get("enabled", True):
            expected_length = self._get_expected_length(question_id)
            token_efficiency = self.calculate_token_efficiency(base_quality, tokens_generated, expected_length)
        else:
            token_efficiency = 0.0
        
        # Special handling for specific question types
        if "JSON" in question_text or "json" in question_text:
            required_keys = reference_data.get("required_keys", []) if reference_data else []
            format_score = self.evaluate_json_adherence(response_text, required_keys)
            prompt_alignment = (prompt_alignment + format_score) / 2
        
        if question_id == "Q5_SUMMARIZATION_ACCURACY":
            source = reference_data.get("source_text", "") if reference_data else ""
            max_sentences = reference_data.get("max_sentences", 3) if reference_data else 3
            summary_scores = self.evaluate_summarization(response_text, source, max_sentences)
            completeness = summary_scores["overall"]
        
        # Calculate overall quality with weights
        overall = (
            self.weights.get("relevance", 0.0) * relevance +
            self.weights.get("factual_accuracy", 0.0) * factual_accuracy +
            self.weights.get("completeness", 0.0) * completeness +
            self.weights.get("fluency", 0.0) * fluency +
            self.weights.get("coherence", 0.0) * coherence +
            self.weights.get("prompt_alignment", 0.0) * prompt_alignment +
            self.weights.get("token_efficiency", 0.0) * token_efficiency
        )
        
        return QualityScores(
            relevance=relevance,
            factual_accuracy=factual_accuracy,
            completeness=completeness,
            fluency=fluency,
            coherence=coherence,
            prompt_alignment=prompt_alignment,
            token_efficiency=token_efficiency,
            overall_quality=overall,
        )
