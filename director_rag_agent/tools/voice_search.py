"""
Local Embedding-based Voice Search System

This module provides functions to:
1. Embed voice profiles from a JSON knowledge base
2. Save embeddings locally for fast retrieval
3. Query voices using semantic search and return exact JSON schema matches
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_KNOWLEDGE_BASE_PATH = "./director_rag_agent/knowledge_bases/eleven_labs_voices_current.json"
DEFAULT_EMBEDDINGS_DIR = "./director_rag_agent/knowledge_bases/voice_embeddings_store"


class VoiceEmbeddingStore:
    """
    A class to manage voice profile embeddings for semantic search.
    """
    
    def __init__(
        self,
        embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the voice embedding store.
        
        Args:
            embeddings_dir: Directory to store embeddings and metadata
            model_name: Sentence transformer model name for embeddings
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.voices: List[Dict[str, Any]] = []
        
        # File paths within the embeddings directory
        self.embeddings_file = self.embeddings_dir / "voice_embeddings.npy"
        self.voices_file = self.embeddings_dir / "voices_metadata.json"
        self.config_file = self.embeddings_dir / "config.json"
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def _create_voice_text(self, voice: Dict[str, Any]) -> str:
        """
        Create a searchable text representation of a voice profile.
        
        Args:
            voice: Voice profile dictionary
            
        Returns:
            Text string for embedding
        """
        parts = []
        
        # Add label/age category
        if "label" in voice:
            parts.append(f"Category: {voice['label']}")
        
        # Add gender
        if "gender" in voice:
            parts.append(f"Gender: {voice['gender']}")
        
        # Add tone
        if "tone" in voice:
            parts.append(f"Tone: {voice['tone']}")
        
        # Add style
        if "style" in voice:
            parts.append(f"Style: {voice['style']}")
        
        # Add name if available
        if "name" in voice:
            parts.append(f"Name: {voice['name']}")
        
        # Add description if available
        if "description" in voice:
            parts.append(f"Description: {voice['description']}")
        
        return ". ".join(parts)
    
    def embed_knowledge_base(
        self,
        knowledge_base_path: str = DEFAULT_KNOWLEDGE_BASE_PATH,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Embed all voice profiles from a JSON knowledge base.
        
        Args:
            knowledge_base_path: Path to the JSON file containing voice profiles
            force_rebuild: If True, rebuild embeddings even if they exist
            
        Returns:
            Dictionary with status and information about the embedding process
        """
        try:
            # Check if embeddings already exist
            if not force_rebuild and self._embeddings_exist():
                logger.info("Embeddings already exist. Use force_rebuild=True to rebuild.")
                return {
                    "status": "exists",
                    "message": "Embeddings already exist. Loading from cache.",
                    "embeddings_dir": str(self.embeddings_dir)
                }
            
            # Load knowledge base
            logger.info(f"Loading knowledge base from: {knowledge_base_path}")
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract voices list
            if isinstance(data, dict) and "data" in data:
                self.voices = data["data"]
            elif isinstance(data, list):
                self.voices = data
            else:
                raise ValueError("Invalid knowledge base format. Expected dict with 'data' key or list.")
            
            logger.info(f"Found {len(self.voices)} voice profiles")
            
            # Create text representations
            voice_texts = [self._create_voice_text(voice) for voice in self.voices]
            
            # Load model and create embeddings
            model = self._load_model()
            logger.info("Creating embeddings...")
            self.embeddings = model.encode(voice_texts, show_progress_bar=True)
            
            # Save embeddings
            self._save_embeddings()
            
            return {
                "status": "success",
                "message": f"Successfully embedded {len(self.voices)} voice profiles",
                "num_voices": len(self.voices),
                "embedding_dim": self.embeddings.shape[1],
                "embeddings_dir": str(self.embeddings_dir)
            }
            
        except FileNotFoundError:
            error_msg = f"Knowledge base not found: {knowledge_base_path}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Failed to embed knowledge base: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def _embeddings_exist(self) -> bool:
        """Check if saved embeddings exist."""
        return (
            self.embeddings_file.exists() and
            self.voices_file.exists() and
            self.config_file.exists()
        )
    
    def _save_embeddings(self) -> None:
        """Save embeddings and metadata to disk."""
        # Create directory if needed
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy array
        np.save(self.embeddings_file, self.embeddings)
        logger.info(f"Saved embeddings to: {self.embeddings_file}")
        
        # Save voice metadata as JSON
        with open(self.voices_file, 'w', encoding='utf-8') as f:
            json.dump(self.voices, f, indent=2)
        logger.info(f"Saved voice metadata to: {self.voices_file}")
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "num_voices": len(self.voices),
            "embedding_dim": int(self.embeddings.shape[1])
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to: {self.config_file}")
    
    def load_embeddings(self) -> Dict[str, Any]:
        """
        Load saved embeddings from disk.
        
        Returns:
            Dictionary with status and loading information
        """
        try:
            if not self._embeddings_exist():
                return {
                    "status": "error",
                    "message": "No saved embeddings found. Run embed_knowledge_base() first."
                }
            
            # Load embeddings
            self.embeddings = np.load(self.embeddings_file)
            logger.info(f"Loaded embeddings: shape {self.embeddings.shape}")
            
            # Load voice metadata
            with open(self.voices_file, 'r', encoding='utf-8') as f:
                self.voices = json.load(f)
            logger.info(f"Loaded {len(self.voices)} voice profiles")
            
            # Load and verify config
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update model name from config
            self.model_name = config.get("model_name", self.model_name)
            
            return {
                "status": "success",
                "message": f"Loaded {len(self.voices)} voice embeddings",
                "num_voices": len(self.voices),
                "embedding_dim": self.embeddings.shape[1]
            }
            
        except Exception as e:
            error_msg = f"Failed to load embeddings: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def query_voices(
        self,
        query: str,
        top_k: int = 3,
        gender_filter: Optional[str] = None,
        tone_filter: Optional[str] = None,
        style_filter: Optional[str] = None,
        label_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for voice profiles using semantic similarity.
        
        Args:
            query: Search query describing the desired voice
            top_k: Number of results to return (default 5)
            gender_filter: Optional filter for gender (e.g., "Male", "Female")
            tone_filter: Optional filter for tone (e.g., "Calm", "Energetic")
            style_filter: Optional filter for style (e.g., "American", "British")
            label_filter: Optional filter for label/age category
            
        Returns:
            Dictionary containing:
            - status: "success" or "error"
            - voices: List of matching voice profiles in original JSON schema
            - scores: Similarity scores for each result
            - query: The original query
        """
        try:
            # Ensure embeddings are loaded
            if self.embeddings is None or len(self.voices) == 0:
                load_result = self.load_embeddings()
                if load_result["status"] == "error":
                    return {
                        "status": "error",
                        "message": load_result["message"],
                        "voices": [],
                        "scores": [],
                        "query": query
                    }
            
            # Try filters with progressive relaxation
            filtered_indices, applied_filters = self._apply_filters_with_fallback(
                gender_filter, tone_filter, style_filter, label_filter
            )
            
            # Embed the query
            model = self._load_model()
            query_embedding = model.encode([query])[0]
            
            # Get filtered embeddings
            filtered_embeddings = self.embeddings[filtered_indices]
            
            # Calculate cosine similarity
            similarities = self._cosine_similarity(query_embedding, filtered_embeddings)
            
            # Get top-k indices within filtered set
            top_k = 3
            top_k = min(top_k, len(filtered_indices))
            top_local_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Map back to original indices
            top_original_indices = [filtered_indices[i] for i in top_local_indices]
            top_scores = [float(similarities[i]) for i in top_local_indices]
            
            # Get voice profiles in original schema
            result_voices = [self.voices[i] for i in top_original_indices]
            
            # Build message indicating any relaxed filters
            message = f"Found {len(result_voices)} matching voices"
            if applied_filters.get("relaxed"):
                message += f" (relaxed filters: {', '.join(applied_filters['relaxed'])})"
            
            return {
                "status": "success",
                "voices": result_voices,
                "scores": top_scores,
                "query": query,
                "filters_applied": applied_filters,
                "message": message
            }
            
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "voices": [],
                "scores": [],
                "query": query
            }

    def _apply_filters_with_fallback(
        self,
        gender_filter: Optional[str],
        tone_filter: Optional[str],
        style_filter: Optional[str],
        label_filter: Optional[str]
    ) -> tuple[List[int], Dict[str, Any]]:
        """
        Apply filters with progressive relaxation to ensure at least 1 result.
        
        Relaxation order (least to most important):
        1. tone (most flexible - rely on semantic search)
        2. style (accent is secondary)
        3. label (age category)
        4. gender (most important to preserve)
        
        Returns:
            Tuple of (matching indices, dict of applied/relaxed filters)
        """
        filters = [
            ("tone", tone_filter),
            ("style", style_filter),
            ("label", label_filter),
            ("gender", gender_filter),
        ]
        
        # Start with all filters
        active_filters = {name: value for name, value in filters if value is not None}
        relaxed_filters = []
        
        # Try progressively relaxing filters until we get results
        filters_to_try = list(active_filters.keys())
        
        while True:
            indices = self._apply_filters(
                gender_filter=active_filters.get("gender"),
                tone_filter=active_filters.get("tone"),
                style_filter=active_filters.get("style"),
                label_filter=active_filters.get("label")
            )
            
            if len(indices) > 0:
                break
            
            # No results - relax the first remaining filter
            if filters_to_try:
                filter_to_relax = filters_to_try.pop(0)
                if filter_to_relax in active_filters:
                    relaxed_filters.append(f"{filter_to_relax}={active_filters[filter_to_relax]}")
                    del active_filters[filter_to_relax]
                    logger.info(f"Relaxing filter: {filter_to_relax}")
            else:
                # All filters relaxed, return all indices
                indices = list(range(len(self.voices)))
                break
        
        applied_info = {
            "applied": active_filters,
            "relaxed": relaxed_filters if relaxed_filters else None
        }
        
        return indices, applied_info

    def _apply_filters(
        self,
        gender_filter: Optional[str],
        tone_filter: Optional[str],
        style_filter: Optional[str],
        label_filter: Optional[str]
    ) -> List[int]:
        """Apply filters and return indices of matching voices."""
        indices = []
        
        for i, voice in enumerate(self.voices):
            # Check gender filter
            if gender_filter and voice.get("gender", "").lower() != gender_filter.lower():
                continue
            
            # Check tone filter (partial match)
            if tone_filter and tone_filter.lower() not in voice.get("tone", "").lower():
                continue
            
            # Check style filter (partial match)
            if style_filter and style_filter.lower() not in voice.get("style", "").lower():
                continue
            
            # Check label filter (partial match)
            if label_filter and label_filter.lower() not in voice.get("label", "").lower():
                continue
            
            indices.append(i)
        
        return indices
    
    def _cosine_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between query and embeddings."""
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate dot product
        return np.dot(embeddings_norm, query_norm)


# Convenience functions for direct use

def embed_voices(
    knowledge_base_path: str = DEFAULT_KNOWLEDGE_BASE_PATH,
    embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Embed voice profiles from a knowledge base and save locally.
    
    Args:
        knowledge_base_path: Path to the JSON knowledge base
        embeddings_dir: Directory to save embeddings
        force_rebuild: If True, rebuild even if embeddings exist
        
    Returns:
        Status dictionary with embedding results
    """
    store = VoiceEmbeddingStore(embeddings_dir=embeddings_dir)
    return store.embed_knowledge_base(knowledge_base_path, force_rebuild)


store = VoiceEmbeddingStore(embeddings_dir=DEFAULT_EMBEDDINGS_DIR)

def search_voices(
    query: str,
    gender: Optional[str] = None,
    tone: Optional[str] = None,
    style: Optional[str] = None,
    label: Optional[str] = None,
    top_k: int = 3,
    embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR
) -> Dict[str, Any]:
    """
    Search for voice profiles using semantic search.
    
    Args:
        query: Search query describing the desired voice
        top_k: Number of results to return (default 5)
        gender: Optional gender filter ("Male", "Female")
        tone: Optional tone filter ("Calm", "Energetic", etc.)
        style: Optional style filter ("American", "British", etc.)
        label: Optional label/age filter ("Older Male", "Female Teen", etc.)
        embeddings_dir: Directory containing saved embeddings
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - voices: List of voice profiles in original JSON schema
        - scores: Similarity scores
        - query: Original query
    """
    return store.query_voices(
        query=query,
        top_k=top_k,
        gender_filter=gender,
        tone_filter=tone,
        style_filter=style,
        label_filter=label
    )


def load_voice_store(embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR) -> Dict[str, Any]:
    """
    Load an existing voice embedding store.
    
    Args:
        embeddings_dir: Directory containing saved embeddings
        
    Returns:
        Status dictionary with loading results
    """
    store = VoiceEmbeddingStore(embeddings_dir=embeddings_dir)
    return store.load_embeddings()


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    search_result = search_voices(
        query="charming and playful young male soldier",
        top_k=5,
        gender="Male",
        tone="Playful",
        label= "Male 20s to 30s"
    )
    print(f"   Found {len(search_result['voices'])} voices:")
    for i, (voice, score) in enumerate(zip(search_result['voices'], search_result['scores'])):
        print(f"   {i+1}. {voice['label']} - {voice['tone']} {voice['style']} (score: {score:.3f})")
                
                
    # # Example: Embed the knowledge base
    # print("=" * 60)
    # print("Voice Embedding System - Example Usage")
    # print("=" * 60)
    
    # # Check if knowledge base path was provided
    # kb_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KNOWLEDGE_BASE_PATH
    
    # print(f"\n1. Embedding knowledge base: {kb_path}")
    # result = embed_voices(kb_path, force_rebuild=True)
    # print(f"   Result: {result}")
    
    # if result["status"] in ["success", "exists"]:
    #     print("\n2. Searching for 'deep authoritative male narrator'")
    #     search_result = search_voices(
    #         query="deep authoritative male narrator",
    #         top_k=5
    #     )
    #     print(f"   Found {len(search_result['voices'])} voices:")
    #     for i, (voice, score) in enumerate(zip(search_result['voices'], search_result['scores'])):
    #         print(f"   {i+1}. {voice['label']} - {voice['tone']} {voice['style']} (score: {score:.3f})")
    #         print(f"      voice_id: {voice['voice_id']}")
        
    #     print("\n3. Searching for 'young female energetic' with gender filter")
    #     search_result = search_voices(
    #         query="young energetic bubbly",
    #         top_k=5,
    #         gender="Female"
    #     )
    #     print(f"   Found {len(search_result['voices'])} voices:")
    #     for i, (voice, score) in enumerate(zip(search_result['voices'], search_result['scores'])):
    #         print(f"   {i+1}. {voice['label']} - {voice['tone']} {voice['style']} (score: {score:.3f})")
    
  