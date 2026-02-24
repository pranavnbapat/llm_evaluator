"""Scientific evaluation metrics for LLM responses."""
import re
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from collections import Counter

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
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


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
        """Lazy load the embedding model."""
        if self.model is None and ML_AVAILABLE and self.model_name:
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        model = self.load()
        if model is None:
            return np.zeros((len(texts), 768))  # Return dummy embeddings
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class ResponseEvaluator:
    """Scientific evaluator for LLM responses."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self._embeddings_cache = {}
    
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
    
    def calculate_fluency(self, response: str, expected_language: str = None) -> float:
        """
        Calculate fluency score based on multiple factors.
        """
        if not response.strip():
            return 0.0
        
        scores = []
        
        # 1. Sentence structure (basic check)
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check average sentence length (too long or too short is bad)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        if 5 <= avg_sentence_length <= 30:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # 2. Repetition penalty
        words = response.lower().split()
        if words:
            word_counts = Counter(words)
            max_repetition = max(word_counts.values())
            repetition_score = 1.0 - (max_repetition / len(words))
            scores.append(max(0.0, repetition_score))
        
        # 3. Language detection (if available)
        if LANGDETECT_AVAILABLE and expected_language:
            try:
                detected = langdetect.detect(response)
                lang_score = 1.0 if detected == expected_language.lower() else 0.5
                scores.append(lang_score)
            except:
                pass
        
        return np.mean(scores) if scores else 0.5
    
    def calculate_coherence(self, response: str) -> float:
        """
        Calculate coherence score based on discourse flow.
        """
        if not response.strip():
            return 0.0
        
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.7  # Single sentence is moderately coherent
        
        # Check for transition words
        transition_words = [
            "however", "therefore", "furthermore", "moreover", "additionally",
            "consequently", "nevertheless", "meanwhile", "subsequently",
            "first", "second", "third", "finally", "in conclusion",
            "for example", "such as", "in particular", "specifically"
        ]
        
        transition_count = sum(
            1 for tw in transition_words
            if tw in response.lower()
        )
        
        # Score based on transition word density
        transition_score = min(1.0, transition_count / max(1, len(sentences) / 3))
        
        # Check for consistent entity mentions (simple version)
        entity_coherence = 0.5  # Default
        
        return (transition_score + entity_coherence) / 2
    
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
        
        # Check if response is too generic
        generic_phrases = [
            "i cannot answer", "i don't know", "i'm not sure",
            "as an ai", "as a language model", "i apologize"
        ]
        
        response_lower = response.lower()
        for phrase in generic_phrases:
            if phrase in response_lower:
                return 0.3  # Penalty for generic refusals
        
        # Check relevance to question topic
        relevance = self.calculate_relevance(question, response)
        
        return relevance
    
    def calculate_token_efficiency(
        self,
        quality_score: float,
        token_count: int,
        expected_length: Tuple[int, int] = (50, 500),
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
        
        # Calculate individual scores
        relevance = self.calculate_relevance(question_text, response_text)
        
        completeness = self.calculate_completeness(response_text, expected_elements)
        
        factual_accuracy = self.calculate_factual_accuracy(response_text, reference_facts)
        
        fluency = self.calculate_fluency(response_text, language)
        
        coherence = self.calculate_coherence(response_text)
        
        prompt_alignment = self.calculate_prompt_alignment(response_text, question_text)
        
        # Composite quality before efficiency
        base_quality = (relevance + factual_accuracy + completeness + fluency + coherence + prompt_alignment) / 6
        token_efficiency = self.calculate_token_efficiency(base_quality, tokens_generated)
        
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
        from app.config import settings
        weights = settings.composite_weights
        
        overall = (
            weights["relevance"] * relevance +
            weights["factual_accuracy"] * factual_accuracy +
            weights["completeness"] * completeness +
            weights["fluency"] * fluency +
            weights["coherence"] * coherence +
            weights["prompt_alignment"] * prompt_alignment +
            weights["token_efficiency"] * token_efficiency
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
