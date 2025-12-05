"""
Local Embedding-based Audio Search System

This module provides a unified, reusable system for:
1. Embedding audio catalogs (music, ambience, sfx) from JSON knowledge bases
2. Saving embeddings locally for fast retrieval
3. Querying audio using semantic search with filters

Supports: Music, Ambience, SFX catalogs with different schema fields.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class AudioType(Enum):
    """Types of audio catalogs supported."""
    MUSIC = "music"
    AMBIENCE = "ambience"
    SFX = "sfx"
    VOICE = "voice"


@dataclass
class AudioCatalogConfig:
    """Configuration for different audio catalog types."""
    audio_type: AudioType
    knowledge_base_path: str
    embeddings_dir: str
    # Fields to include in the searchable text
    text_fields: List[str] = field(default_factory=list)
    # Fields that can be used as filters
    filter_fields: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Default text fields based on audio type
        if not self.text_fields:
            self.text_fields = self._default_text_fields()
        if not self.filter_fields:
            self.filter_fields = self._default_filter_fields()
    
    def _default_text_fields(self) -> List[str]:
        """Default fields for creating searchable text."""
        defaults = {
            AudioType.MUSIC: ["title", "group", "mood", "style", "tags"],
            AudioType.AMBIENCE: ["group", "mood", "style", "tags"],
            AudioType.SFX: ["group", "mood", "style", "tags"],
            AudioType.VOICE: ["label", "gender", "tone", "style", "name", "description"],
        }
        return defaults.get(self.audio_type, ["title", "group", "mood", "style", "tags"])
    
    def _default_filter_fields(self) -> List[str]:
        """Default filterable fields."""
        defaults = {
            AudioType.MUSIC: ["group", "style", "intensity", "loopable"],
            AudioType.AMBIENCE: ["group", "style", "intensity", "loopable"],
            AudioType.SFX: ["group", "style", "intensity", "loopable"],
            AudioType.VOICE: ["gender", "tone", "style", "label"],
        }
        return defaults.get(self.audio_type, ["group", "style"])


class AudioEmbeddingStore:
    """
    A unified class to manage audio catalog embeddings for semantic search.
    Works with music, ambience, sfx, and voice catalogs.
    """
    
    def __init__(
        self,
        config: AudioCatalogConfig,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the audio embedding store.
        
        Args:
            config: Configuration for the audio catalog
            model_name: Sentence transformer model name for embeddings
        """
        self.config = config
        self.embeddings_dir = Path(config.embeddings_dir)
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.items: List[Dict[str, Any]] = []
        
        # File paths within the embeddings directory
        self.embeddings_file = self.embeddings_dir / f"{config.audio_type.value}_embeddings.npy"
        self.items_file = self.embeddings_dir / f"{config.audio_type.value}_metadata.json"
        self.config_file = self.embeddings_dir / f"{config.audio_type.value}_config.json"
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def _create_searchable_text(self, item: Dict[str, Any]) -> str:
        """
        Create a searchable text representation of an audio item.
        
        Args:
            item: Audio item dictionary
            
        Returns:
            Text string for embedding
        """
        parts = []
        
        for field_name in self.config.text_fields:
            if field_name in item:
                value = item[field_name]
                # Handle different value types
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                
                # Create descriptive text
                parts.append(f"{field_name.replace('_', ' ').title()}: {value_str}")
        
        return ". ".join(parts)
    
    def embed_knowledge_base(
        self,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Embed all items from a JSON knowledge base.
        
        Args:
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
            kb_path = self.config.knowledge_base_path
            logger.info(f"Loading knowledge base from: {kb_path}")
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract items list
            if isinstance(data, dict) and "data" in data:
                self.items = data["data"]
            elif isinstance(data, list):
                self.items = data
            else:
                raise ValueError("Invalid knowledge base format. Expected dict with 'data' key or list.")
            
            logger.info(f"Found {len(self.items)} {self.config.audio_type.value} items")
            
            # Create text representations
            item_texts = [self._create_searchable_text(item) for item in self.items]
            
            # Load model and create embeddings
            model = self._load_model()
            logger.info("Creating embeddings...")
            self.embeddings = model.encode(item_texts, show_progress_bar=True)
            
            # Save embeddings
            self._save_embeddings()
            
            return {
                "status": "success",
                "message": f"Successfully embedded {len(self.items)} {self.config.audio_type.value} items",
                "num_items": len(self.items),
                "embedding_dim": self.embeddings.shape[1],
                "embeddings_dir": str(self.embeddings_dir)
            }
            
        except FileNotFoundError:
            error_msg = f"Knowledge base not found: {self.config.knowledge_base_path}"
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
            self.items_file.exists() and
            self.config_file.exists()
        )
    
    def _save_embeddings(self) -> None:
        """Save embeddings and metadata to disk."""
        # Create directory if needed
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy array
        np.save(self.embeddings_file, self.embeddings)
        logger.info(f"Saved embeddings to: {self.embeddings_file}")
        
        # Save item metadata as JSON
        with open(self.items_file, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, indent=2)
        logger.info(f"Saved item metadata to: {self.items_file}")
        
        # Save configuration
        config_data = {
            "model_name": self.model_name,
            "audio_type": self.config.audio_type.value,
            "num_items": len(self.items),
            "embedding_dim": int(self.embeddings.shape[1]),
            "text_fields": self.config.text_fields,
            "filter_fields": self.config.filter_fields
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
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
            
            # Load item metadata
            with open(self.items_file, 'r', encoding='utf-8') as f:
                self.items = json.load(f)
            logger.info(f"Loaded {len(self.items)} {self.config.audio_type.value} items")
            
            # Load and verify config
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update model name from config
            self.model_name = config_data.get("model_name", self.model_name)
            
            return {
                "status": "success",
                "message": f"Loaded {len(self.items)} {self.config.audio_type.value} embeddings",
                "num_items": len(self.items),
                "embedding_dim": self.embeddings.shape[1]
            }
            
        except Exception as e:
            error_msg = f"Failed to load embeddings: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def query(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for audio items using semantic similarity.
        
        Args:
            query: Search query describing the desired audio
            top_k: Number of results to return
            filters: Optional dict of field_name -> value for filtering
                     Supports: exact match for strings/bools, range for ints (use tuple)
            
        Returns:
            Dictionary containing:
            - status: "success" or "error"
            - items: List of matching items in original JSON schema
            - scores: Similarity scores for each result
            - query: The original query
        """
        try:
            # Ensure embeddings are loaded
            if self.embeddings is None or len(self.items) == 0:
                load_result = self.load_embeddings()
                if load_result["status"] == "error":
                    return {
                        "status": "error",
                        "message": load_result["message"],
                        "items": [],
                        "scores": [],
                        "query": query
                    }
            
            # Apply filters with progressive relaxation
            filtered_indices, applied_filters = self._apply_filters_with_fallback(filters or {})
            
            # Embed the query
            model = self._load_model()
            query_embedding = model.encode([query])[0]
            
            # Get filtered embeddings
            filtered_embeddings = self.embeddings[filtered_indices]
            
            # Calculate cosine similarity
            similarities = self._cosine_similarity(query_embedding, filtered_embeddings)
            
            # Get top-k indices within filtered set
            actual_top_k = min(top_k, len(filtered_indices))
            top_local_indices = np.argsort(similarities)[-actual_top_k:][::-1]
            
            # Map back to original indices
            top_original_indices = [filtered_indices[i] for i in top_local_indices]
            top_scores = [float(similarities[i]) for i in top_local_indices]
            
            # Get items in original schema
            result_items = [self.items[i] for i in top_original_indices]
            
            # Build message indicating any relaxed filters
            message = f"Found {len(result_items)} matching {self.config.audio_type.value} items"
            if applied_filters.get("relaxed"):
                message += f" (relaxed filters: {', '.join(applied_filters['relaxed'])})"
            
            return {
                "status": "success",
                "items": result_items,
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
                "items": [],
                "scores": [],
                "query": query
            }
    
    def _apply_filters_with_fallback(
        self,
        filters: Dict[str, Any]
    ) -> tuple:
        """
        Apply filters with progressive relaxation to ensure at least 1 result.
        
        Returns:
            Tuple of (matching indices, dict of applied/relaxed filters)
        """
        if not filters:
            return list(range(len(self.items))), {"applied": {}, "relaxed": None}
        
        # Order filters by importance (less important first for relaxation)
        # style/mood are usually less critical than group/intensity
        filter_priority = ["style", "mood", "loopable", "intensity", "group"]
        
        active_filters = dict(filters)
        relaxed_filters = []
        
        # Sort filters by priority (least important first)
        sorted_filter_keys = sorted(
            active_filters.keys(),
            key=lambda x: filter_priority.index(x) if x in filter_priority else len(filter_priority)
        )
        
        while True:
            indices = self._apply_filters(active_filters)
            
            if len(indices) > 0:
                break
            
            # No results - relax the first remaining filter
            if sorted_filter_keys:
                filter_to_relax = sorted_filter_keys.pop(0)
                if filter_to_relax in active_filters:
                    relaxed_filters.append(f"{filter_to_relax}={active_filters[filter_to_relax]}")
                    del active_filters[filter_to_relax]
                    logger.info(f"Relaxing filter: {filter_to_relax}")
            else:
                # All filters relaxed, return all indices
                indices = list(range(len(self.items)))
                break
        
        applied_info = {
            "applied": active_filters,
            "relaxed": relaxed_filters if relaxed_filters else None
        }
        
        return indices, applied_info
    
    def _apply_filters(self, filters: Dict[str, Any]) -> List[int]:
        """Apply filters and return indices of matching items."""
        indices = []
        
        for i, item in enumerate(self.items):
            match = True
            
            for field_name, filter_value in filters.items():
                item_value = item.get(field_name)
                
                if item_value is None:
                    match = False
                    break
                
                # Handle different filter types
                if isinstance(filter_value, tuple) and len(filter_value) == 2:
                    # Range filter for numeric fields (min, max)
                    min_val, max_val = filter_value
                    if not (min_val <= item_value <= max_val):
                        match = False
                        break
                elif isinstance(filter_value, bool):
                    # Exact match for boolean
                    if item_value != filter_value:
                        match = False
                        break
                elif isinstance(item_value, list):
                    # Check if filter value is in the list (for mood, tags, etc.)
                    if isinstance(filter_value, str):
                        if filter_value.lower() not in [str(v).lower() for v in item_value]:
                            match = False
                            break
                    elif isinstance(filter_value, list):
                        # Check if any filter value is in item value
                        if not any(fv.lower() in [str(v).lower() for v in item_value] for fv in filter_value):
                            match = False
                            break
                else:
                    # Partial string match (case-insensitive)
                    if str(filter_value).lower() not in str(item_value).lower():
                        match = False
                        break
            
            if match:
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


# =============================================================================
# Pre-configured stores and convenience functions
# =============================================================================

# Default paths - adjust these to your actual paths
DEFAULT_BASE_PATH = "./director_rag_agent/knowledge_bases"
DEFAULT_EMBEDDINGS_BASE = "./director_rag_agent/knowledge_bases/audio_embeddings_store"

# Catalog configurations
MUSIC_CONFIG = AudioCatalogConfig(
    audio_type=AudioType.MUSIC,
    knowledge_base_path=f"{DEFAULT_BASE_PATH}/music_catalog_full.json",
    embeddings_dir=f"{DEFAULT_EMBEDDINGS_BASE}/music"
)

AMBIENCE_CONFIG = AudioCatalogConfig(
    audio_type=AudioType.AMBIENCE,
    knowledge_base_path=f"{DEFAULT_BASE_PATH}/ambience_catalog_enriched.json",
    embeddings_dir=f"{DEFAULT_EMBEDDINGS_BASE}/ambience"
)

SFX_CONFIG = AudioCatalogConfig(
    audio_type=AudioType.SFX,
    knowledge_base_path=f"{DEFAULT_BASE_PATH}/sfx_catalog_enriched.json",
    embeddings_dir=f"{DEFAULT_EMBEDDINGS_BASE}/sfx"
)

# Initialize stores (lazy loading)
_music_store: Optional[AudioEmbeddingStore] = None
_ambience_store: Optional[AudioEmbeddingStore] = None
_sfx_store: Optional[AudioEmbeddingStore] = None


def _get_music_store() -> AudioEmbeddingStore:
    global _music_store
    if _music_store is None:
        _music_store = AudioEmbeddingStore(config=MUSIC_CONFIG)
    return _music_store


def _get_ambience_store() -> AudioEmbeddingStore:
    global _ambience_store
    if _ambience_store is None:
        _ambience_store = AudioEmbeddingStore(config=AMBIENCE_CONFIG)
    return _ambience_store


def _get_sfx_store() -> AudioEmbeddingStore:
    global _sfx_store
    if _sfx_store is None:
        _sfx_store = AudioEmbeddingStore(config=SFX_CONFIG)
    return _sfx_store


# =============================================================================
# Public API Functions
# =============================================================================

def embed_music(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Embed music catalog and save locally.
    
    Args:
        force_rebuild: If True, rebuild even if embeddings exist
        
    Returns:
        Status dictionary with embedding results
    """
    return _get_music_store().embed_knowledge_base(force_rebuild=force_rebuild)


def embed_ambience(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Embed ambience catalog and save locally.
    
    Args:
        force_rebuild: If True, rebuild even if embeddings exist
        
    Returns:
        Status dictionary with embedding results
    """
    return _get_ambience_store().embed_knowledge_base(force_rebuild=force_rebuild)


def embed_sfx(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Embed sfx catalog and save locally.
    
    Args:
        force_rebuild: If True, rebuild even if embeddings exist
        
    Returns:
        Status dictionary with embedding results
    """
    return _get_sfx_store().embed_knowledge_base(force_rebuild=force_rebuild)


def embed_all(force_rebuild: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Embed all audio catalogs (music, ambience, sfx).
    
    Args:
        force_rebuild: If True, rebuild even if embeddings exist
        
    Returns:
        Dictionary with results for each catalog type
    """
    return {
        "music": embed_music(force_rebuild),
        "ambience": embed_ambience(force_rebuild),
        "sfx": embed_sfx(force_rebuild)
    }


def search_music(
    query: str,
    top_k: int = 3,
    group: Optional[str] = None,
    style: Optional[str] = None,
    mood: Optional[str] = None,
    intensity: Optional[tuple] = None,
    loopable: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Search for music using semantic search.
    
    Args:
        query: Search query describing the desired music
        top_k: Number of results to return (default 3)
        group: Optional filter for group (e.g., "EPIC", "UPLIFTING")
        style: Optional filter for style (e.g., "orchestral strings", "electronic")
        mood: Optional filter for mood (e.g., "epic", "peaceful")
        intensity: Optional tuple (min, max) for intensity filter (0-100)
        loopable: Optional filter for loopable tracks
        
    Returns:
        Dictionary with status, items, scores, and query info
    """
    filters = {}
    if group:
        filters["group"] = group
    if style:
        filters["style"] = style
    if mood:
        filters["mood"] = mood
    if intensity:
        filters["intensity"] = intensity
    if loopable is not None:
        filters["loopable"] = loopable
    
    return _get_music_store().query(query=query, top_k=top_k, filters=filters)


def search_ambience(
    query: str,
    top_k: int = 3,
    group: Optional[str] = None,
    style: Optional[str] = None,
    mood: Optional[str] = None,
    intensity: Optional[tuple] = None,
    loopable: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Search for ambience sounds using semantic search.
    
    Args:
        query: Search query describing the desired ambience
        top_k: Number of results to return (default 3)
        group: Optional filter for group (e.g., "People in the City", "Suburbs & Countryside")
        style: Optional filter for style
        mood: Optional filter for mood (e.g., "peaceful", "busy")
        intensity: Optional tuple (min, max) for intensity filter (0-100)
        loopable: Optional filter for loopable tracks
        
    Returns:
        Dictionary with status, items, scores, and query info
    """
    filters = {}
    if group:
        filters["group"] = group
    if style:
        filters["style"] = style
    if mood:
        filters["mood"] = mood
    if intensity:
        filters["intensity"] = intensity
    if loopable is not None:
        filters["loopable"] = loopable
    
    return _get_ambience_store().query(query=query, top_k=top_k, filters=filters)


def search_sfx(
    query: str,
    top_k: int = 3,
    group: Optional[str] = None,
    style: Optional[str] = None,
    mood: Optional[str] = None,
    intensity: Optional[tuple] = None,
    loopable: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Search for sound effects using semantic search.
    
    Args:
        query: Search query describing the desired sound effect
        top_k: Number of results to return (default 3)
        group: Optional filter for group (e.g., "Fighting", "Explosions")
        style: Optional filter for style
        mood: Optional filter for mood
        intensity: Optional tuple (min, max) for intensity filter (0-100)
        loopable: Optional filter for loopable tracks
        
    Returns:
        Dictionary with status, items, scores, and query info
    """
    filters = {}
    if group:
        filters["group"] = group
    if style:
        filters["style"] = style
    if mood:
        filters["mood"] = mood
    if intensity:
        filters["intensity"] = intensity
    if loopable is not None:
        filters["loopable"] = loopable
    
    return _get_sfx_store().query(query=query, top_k=top_k, filters=filters)


def create_custom_store(
    audio_type: str,
    knowledge_base_path: str,
    embeddings_dir: str,
    text_fields: Optional[List[str]] = None,
    filter_fields: Optional[List[str]] = None
) -> AudioEmbeddingStore:
    """
    Create a custom audio embedding store with custom configuration.
    
    Args:
        audio_type: Type identifier (e.g., "music", "ambience", "sfx", "voice")
        knowledge_base_path: Path to JSON knowledge base
        embeddings_dir: Directory to store embeddings
        text_fields: Fields to include in searchable text
        filter_fields: Fields that can be used as filters
        
    Returns:
        Configured AudioEmbeddingStore instance
    """
    # Map string to AudioType enum
    type_map = {
        "music": AudioType.MUSIC,
        "ambience": AudioType.AMBIENCE,
        "sfx": AudioType.SFX,
        "voice": AudioType.VOICE
    }
    
    audio_type_enum = type_map.get(audio_type.lower(), AudioType.MUSIC)
    
    config = AudioCatalogConfig(
        audio_type=audio_type_enum,
        knowledge_base_path=knowledge_base_path,
        embeddings_dir=embeddings_dir,
        text_fields=text_fields or [],
        filter_fields=filter_fields or []
    )
    
    return AudioEmbeddingStore(config=config)


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Audio Embedding System - Example Usage")
    print("=" * 60)
    
    # Example 1: Embed and search music
    print("\n1. Embedding music catalog...")
    result = embed_music(force_rebuild=True)
    print(f"   Result: {result['status']} - {result.get('message', '')}")
    
    if result["status"] in ["success", "exists"]:
        print("\n2. Searching for 'epic heroic orchestral battle music'")
        search_result = search_music(
            query="epic heroic orchestral battle music",
            top_k=3,
            group="EPIC"
        )
        print(f"search_result: {search_result}")
        
        print(f"   Found {len(search_result['items'])} items:")
        for i, (item, score) in enumerate(zip(search_result['items'], search_result['scores'])):
            print(f"   {i+1}. {item.get('title', item.get('id'))} - {item['group']} (score: {score:.3f})")
            print(f"      Mood: {item.get('mood', [])}")
    
    # Example 2: Embed and search ambience
    print("\n3. Embedding ambience catalog...")
    result = embed_ambience(force_rebuild=True)
    print(f"   Result: {result['status']} - {result.get('message', '')}")
    
    if result["status"] in ["success", "exists"]:
        print("\n4. Searching for 'peaceful nature birds countryside'")
        search_result = search_ambience(
            query="peaceful nature birds countryside",
            top_k=3
        )
        print(f"   Found {len(search_result['items'])} items:")
        for i, (item, score) in enumerate(zip(search_result['items'], search_result['scores'])):
            print(f"   {i+1}. {item.get('id')} - {item['group']} (score: {score:.3f})")
            print(f"      Mood: {item.get('mood', [])}")
    
    # Example 3: Embed and search SFX
    print("\n5. Embedding SFX catalog...")
    result = embed_sfx(force_rebuild=True)
    print(f"   Result: {result['status']} - {result.get('message', '')}")
    
    if result["status"] in ["success", "exists"]:
        print("\n6. Searching for 'intense fighting combat punches'")
        search_result = search_sfx(
            query="intense fighting combat punches",
            top_k=3,
            group="Fighting"
        )
        print(f"   Found {len(search_result['items'])} items:")
        for i, (item, score) in enumerate(zip(search_result['items'], search_result['scores'])):
            print(f"   {i+1}. {item.get('id')} - {item['group']} (score: {score:.3f})")
            print(f"      Intensity: {item.get('intensity')}")