"""
Voice Search Tool for Director Agent

Provides search_voices function for the Director agent
to find appropriate voice profiles for characters.
"""

import os
from typing import Dict, Any, Optional

# Default embeddings directory
DEFAULT_EMBEDDINGS_DIR = os.getenv("VOICE_EMBEDDINGS_DIR", "./director_rag_agent/knowledge_bases/voice_embeddings_store")

# Try to import the actual voice store
try:
    from director_rag_agent.tools.voice_search import VoiceEmbeddingStore
    store = VoiceEmbeddingStore(embeddings_dir=DEFAULT_EMBEDDINGS_DIR)
except ImportError:
    # Create a stub store if the actual implementation isn't available
    class StubVoiceStore:
        """Stub voice store for when the actual implementation isn't available."""
        
        def query_voices(
            self,
            query: str,
            top_k: int = 3,
            gender_filter: str = None,
            tone_filter: str = None,
            style_filter: str = None,
            label_filter: str = None
        ) -> Dict[str, Any]:
            """Stub implementation for voice search."""
            return {
                "status": "success",
                "voices": [],
                "scores": [],
                "query": query,
                "message": "Voice search stub - implement VoiceStore in voice_store.py"
            }
    
    store = StubVoiceStore()


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
               (e.g., "wise grandfather", "young energetic woman", "narrator documentary")
        top_k: Number of results to return (default 3)
        gender: Optional gender filter ("Male", "Female")
        tone: Optional tone filter. Options include:
              - Calm, Friendly, Southern, Wise, Deep
              - Energetic, Stern, Polished, Caring, Smooth
              - Soft, Breathy, Playful, Bubbly, Raspy
              - Chill, Warm, Casual, Stylish, Comforting
              - Clear and Crisp
        style: Optional style/accent filter. Options include:
               - American, African American, British
               - Southern American, New Yorker, Hispanic American
               - American Midwestern, New Zealand
        label: Optional label/age filter. Options include:
               - "Older Male" - mature male voices
               - "Male 20s to 30s" - young adult male voices
               - "Male Teen" - teenage male voices
               - "Male Under 10" - child male voices
               - "Older Female" - mature female voices
               - "Female - 20s to 30s" - young adult female voices
               - "Female Teen" - teenage female voices
               - "Female Under 10" - child female voices
               - "Narrator" - professional narrator voices (male and female)
        embeddings_dir: Directory containing saved voice embeddings
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - voices: List of voice profiles in original JSON schema
        - scores: Similarity scores
        - query: Original query
        
    Example results item:
        {
            "voice_id": "voice_001",
            "name": "John Smith",
            "gender": "Male",
            "tone": ["Warm", "Friendly"],
            "style": "American",
            "label": "Male 20s to 30s",
            "description": "A warm, friendly male voice suitable for narration"
        }
        
    Voice Matching Strategy:
        1. For narrators: Use label="Narrator" or search "narrator documentary"
        2. For character age: Match label to character age
        3. For personality: Combine query + tone filter
        4. For regional characters: Use style filter
    """
    return store.query_voices(
        query=query,
        top_k=top_k,
        gender_filter=gender,
        tone_filter=tone,
        style_filter=style,
        label_filter=label
    )


# Example voice profiles for testing (remove in production)
EXAMPLE_VOICES = [
    {
        "voice_id": "narrator_male_01",
        "name": "Documentary Narrator",
        "gender": "Male",
        "tone": ["Calm", "Clear and Crisp"],
        "style": "American",
        "label": "Narrator",
        "description": "Professional male narrator voice, warm and authoritative"
    },
    {
        "voice_id": "narrator_female_01",
        "name": "Audiobook Narrator",
        "gender": "Female",
        "tone": ["Warm", "Smooth"],
        "style": "American",
        "label": "Narrator",
        "description": "Professional female narrator voice, engaging and versatile"
    },
    {
        "voice_id": "male_young_01",
        "name": "Young Hero",
        "gender": "Male",
        "tone": ["Energetic", "Friendly"],
        "style": "American",
        "label": "Male 20s to 30s",
        "description": "Young adult male voice, energetic and likable"
    },
    {
        "voice_id": "female_young_01",
        "name": "Young Professional",
        "gender": "Female",
        "tone": ["Polished", "Confident"],
        "style": "American",
        "label": "Female - 20s to 30s",
        "description": "Young adult female voice, professional and confident"
    },
    {
        "voice_id": "male_older_01",
        "name": "Wise Mentor",
        "gender": "Male",
        "tone": ["Wise", "Deep"],
        "style": "American",
        "label": "Older Male",
        "description": "Mature male voice with gravitas and wisdom"
    },
    {
        "voice_id": "female_older_01",
        "name": "Caring Mother",
        "gender": "Female",
        "tone": ["Caring", "Warm"],
        "style": "American",
        "label": "Older Female",
        "description": "Mature female voice, warm and nurturing"
    }
]
