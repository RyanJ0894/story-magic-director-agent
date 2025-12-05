# # Copyright 2025
# # Search Tools for Director Agent - Query Knowledge Bases

# """
# Search Tools - Search through knowledge base files for:
# - Voice profiles (ElevenLabs voices)
# - Music tracks
# - Ambience sounds
# - Sound effects (SFX)

# These tools search through JSON/document knowledge bases to find
# appropriate assets for audio production.

# NOTE: Google ADK does not support Optional/Union types in function parameters.
# All parameters must be simple types (str, int, float, bool, list, dict).
# Using empty string "" as default instead of None for optional string parameters.
# """

# import os
# import json
# import logging
# from pathlib import Path
# from typing import Dict, Any, List
# from google.adk.tools import FunctionTool

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # --- Knowledge Base Paths ---
# # These should point to your actual knowledge base files
# # Update these paths based on your setup

# KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_bases"

# VOICES_KB_PATH = KNOWLEDGE_BASE_DIR / "eleven_labs_voices_current.json"
# MUSIC_KB_PATH = KNOWLEDGE_BASE_DIR / "music_catalog_full.json"
# AMBIENCE_KB_PATH = KNOWLEDGE_BASE_DIR / "ambience_catalog_enriched.json"
# SFX_KB_PATH = KNOWLEDGE_BASE_DIR / "sfx_catalog_enriched.json"


# def load_knowledge_base(kb_path: Path) -> List[Dict]:
#     """
#     Load a knowledge base JSON file.
    
#     Args:
#         kb_path: Path to the JSON file
        
#     Returns:
#         List of dictionaries containing the knowledge base data,
#         or empty list if file not found
#     """
#     if not kb_path.exists():
#         logger.warning(f"Knowledge base not found: {kb_path}")
#         return []
    
#     try:
#         with open(kb_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             # Handle both list and dict formats
#             if isinstance(data, list):
#                 return data
#             elif isinstance(data, dict):
#                 # If it's a dict with a 'data' or 'items' key
#                 if 'data' in data:
#                     return data['data']
#                 elif 'items' in data:
#                     return data['items']
#                 elif 'voices' in data:
#                     return data['voices']
#                 elif 'tracks' in data:
#                     return data['tracks']
#                 else:
#                     # Return as single-item list
#                     return [data]
#             return data
#     except Exception as e:
#         logger.error(f"Error loading knowledge base {kb_path}: {e}")
#         return []


# def fuzzy_match(query: str, text: str) -> bool:
#     """
#     Simple fuzzy matching - checks if any query word is in the text.
    
#     Args:
#         query: Search query string
#         text: Text to search in
        
#     Returns:
#         True if any query word matches
#     """
#     if not query or not text:
#         return False
    
#     query_words = query.lower().split()
#     text_lower = text.lower()
    
#     # Check if any query word is in the text
#     return any(word in text_lower for word in query_words)


# def search_in_records(
#     records: List[Dict],
#     query: str,
#     search_fields: List[str],
#     max_results: int = 10
# ) -> List[Dict]:
#     """
#     Search through records for matching items.
    
#     Args:
#         records: List of record dictionaries
#         query: Search query
#         search_fields: Fields to search in
#         max_results: Maximum number of results to return
        
#     Returns:
#         List of matching records
#     """
#     if not records:
#         return []
    
#     results = []
#     query_lower = query.lower()
    
#     for record in records:
#         # Search in specified fields
#         for field in search_fields:
#             if field in record:
#                 value = record[field]
#                 if isinstance(value, str) and fuzzy_match(query, value):
#                     results.append(record)
#                     break
#                 elif isinstance(value, list):
#                     # Search in list fields (like tags, keywords)
#                     for item in value:
#                         if isinstance(item, str) and fuzzy_match(query, item):
#                             results.append(record)
#                             break
        
#         if len(results) >= max_results:
#             break
    
#     return results[:max_results]


# # ============================================================================
# # VOICE SEARCH TOOL
# # ============================================================================

# def search_voices(
#     query: str,
#     gender: str = "",
#     age_range: str = "",
#     tone: str = "",
#     max_results: int = 5,
# ) -> Dict[str, Any]:
#     """
#     Search for ElevenLabs voice profiles.
    
#     Use this tool to find voice profiles categorized by gender, age range,
#     tone, and style, suitable for various character roles and narration.
    
#     Args:
#         query: Search query describing the voice needed (e.g., "young male soldier", 
#                "elderly female narrator", "deep authoritative")
#         gender: Filter for gender - use "male", "female", or "neutral". Leave empty for no filter.
#         age_range: Filter for age - use "young", "middle-aged", or "elderly". Leave empty for no filter.
#         tone: Filter for tone - use "warm", "authoritative", "soft", etc. Leave empty for no filter.
#         max_results: Maximum number of results to return (default 5)
    
#     Returns:
#         A dictionary containing:
#         - status: "success" or "error"
#         - voices: List of matching voice profiles with voice_id, name, description
#         - query: The original search query
#         - message: Description or error message
#     """
#     logger.info(f"[VoiceSearch] Query: '{query}', gender={gender}, age={age_range}, tone={tone}")
    
#     try:
#         # Load knowledge base
#         voices = load_knowledge_base(VOICES_KB_PATH)
        
#         if not voices:
#             return {
#                 "status": "error",
#                 "message": f"Voice knowledge base not found at {VOICES_KB_PATH}. Please ensure the file exists.",
#                 "voices": [],
#                 "query": query
#             }
        
#         # Build combined search query
#         search_query = query
#         if gender:
#             search_query += f" {gender}"
#         if age_range:
#             search_query += f" {age_range}"
#         if tone:
#             search_query += f" {tone}"
        
#         # Search in relevant fields
#         search_fields = [
#             'name', 'voice_id', 'description', 'gender', 
#             'age', 'age_range', 'tone', 'style', 
#             'accent', 'use_case', 'tags', 'keywords'
#         ]
        
#         results = search_in_records(voices, search_query, search_fields, max_results)
        
#         # Format results
#         formatted_results = []
#         for voice in results:
#             formatted_results.append({
#                 "voice_id": voice.get('voice_id') or voice.get('id') or voice.get('name'),
#                 "name": voice.get('name', 'Unknown'),
#                 "gender": voice.get('gender', 'Unknown'),
#                 "age": voice.get('age') or voice.get('age_range', 'Unknown'),
#                 "tone": voice.get('tone') or voice.get('style', 'Unknown'),
#                 "description": voice.get('description', ''),
#                 "use_case": voice.get('use_case', '')
#             })
        
#         return {
#             "status": "success",
#             "voices": formatted_results,
#             "query": query,
#             "total_found": len(formatted_results),
#             "message": f"Found {len(formatted_results)} voice(s) matching '{query}'"
#         }
        
#     except Exception as e:
#         error_msg = f"Voice search failed: {str(e)}"
#         logger.error(f"[VoiceSearch] {error_msg}")
#         return {
#             "status": "error",
#             "message": error_msg,
#             "voices": [],
#             "query": query
#         }


# # ============================================================================
# # MUSIC SEARCH TOOL
# # ============================================================================

# def search_music(
#     query: str,
#     mood: str = "",
#     genre: str = "",
#     style: str = "",
#     max_results: int = 5,
# ) -> Dict[str, Any]:
#     """
#     Search for music tracks by mood, style, and genre.
    
#     Use this tool to find music tracks including orchestral, electronic,
#     jazz, and piano pieces for various emotional contexts.
    
#     Args:
#         query: Search query describing the music needed (e.g., "tense action",
#                "romantic piano", "dramatic orchestral")
#         mood: Filter for mood - use "tense", "romantic", "sad", "uplifting", etc. Leave empty for no filter.
#         genre: Filter for genre - use "orchestral", "electronic", "jazz", etc. Leave empty for no filter.
#         style: Filter for style - use "cinematic", "ambient", "dramatic", etc. Leave empty for no filter.
#         max_results: Maximum number of results to return (default 5)
    
#     Returns:
#         A dictionary containing:
#         - status: "success" or "error"
#         - tracks: List of matching music tracks with track_id, name, mood, genre
#         - query: The original search query
#         - message: Description or error message
#     """
#     logger.info(f"[MusicSearch] Query: '{query}', mood={mood}, genre={genre}, style={style}")
    
#     try:
#         # Load knowledge base
#         tracks = load_knowledge_base(MUSIC_KB_PATH)
        
#         if not tracks:
#             return {
#                 "status": "error",
#                 "message": f"Music knowledge base not found at {MUSIC_KB_PATH}. Please ensure the file exists.",
#                 "tracks": [],
#                 "query": query
#             }
        
#         # Build combined search query
#         search_query = query
#         if mood:
#             search_query += f" {mood}"
#         if genre:
#             search_query += f" {genre}"
#         if style:
#             search_query += f" {style}"
        
#         # Search in relevant fields
#         search_fields = [
#             'name', 'title', 'track_id', 'file_path', 'path',
#             'mood', 'genre', 'style', 'description',
#             'tags', 'keywords', 'emotion', 'category'
#         ]
        
#         results = search_in_records(tracks, search_query, search_fields, max_results)
        
#         # Format results
#         formatted_results = []
#         for track in results:
#             track_id = (
#                 track.get('track_id') or 
#                 track.get('file_path') or 
#                 track.get('path') or 
#                 f"music/{track.get('name', 'unknown')}.mp3"
#             )
#             # Ensure track_id has music/ prefix
#             if not track_id.startswith('music/'):
#                 track_id = f"music/{track_id}"
            
#             formatted_results.append({
#                 "track_id": track_id,
#                 "name": track.get('name') or track.get('title', 'Unknown'),
#                 "mood": track.get('mood', 'Unknown'),
#                 "genre": track.get('genre', 'Unknown'),
#                 "style": track.get('style', 'Unknown'),
#                 "duration": track.get('duration', 'Unknown'),
#                 "description": track.get('description', '')
#             })
        
#         return {
#             "status": "success",
#             "tracks": formatted_results,
#             "query": query,
#             "total_found": len(formatted_results),
#             "message": f"Found {len(formatted_results)} music track(s) matching '{query}'"
#         }
        
#     except Exception as e:
#         error_msg = f"Music search failed: {str(e)}"
#         logger.error(f"[MusicSearch] {error_msg}")
#         return {
#             "status": "error",
#             "message": error_msg,
#             "tracks": [],
#             "query": query
#         }


# # ============================================================================
# # AMBIENCE SEARCH TOOL
# # ============================================================================

# def search_ambience(
#     query: str,
#     environment: str = "",
#     mood: str = "",
#     max_results: int = 5,
# ) -> Dict[str, Any]:
#     """
#     Search for ambient sound effects.
    
#     Use this tool to find ambient sounds including natural environments,
#     urban soundscapes, and crowd noises, categorized by group, style, and mood.
    
#     Args:
#         query: Search query describing the ambience needed (e.g., "forest birds",
#                "city traffic", "hospital busy", "rain storm")
#         environment: Filter for environment type - use "indoor", "outdoor",
#                     "urban", "nature", "industrial", etc. Leave empty for no filter.
#         mood: Filter for mood - use "peaceful", "tense", "busy", etc. Leave empty for no filter.
#         max_results: Maximum number of results to return (default 5)
    
#     Returns:
#         A dictionary containing:
#         - status: "success" or "error"
#         - tracks: List of matching ambience tracks with track_id, name, environment
#         - query: The original search query
#         - message: Description or error message
#     """
#     logger.info(f"[AmbienceSearch] Query: '{query}', environment={environment}, mood={mood}")
    
#     try:
#         # Load knowledge base
#         tracks = load_knowledge_base(AMBIENCE_KB_PATH)
        
#         if not tracks:
#             return {
#                 "status": "error",
#                 "message": f"Ambience knowledge base not found at {AMBIENCE_KB_PATH}. Please ensure the file exists.",
#                 "tracks": [],
#                 "query": query
#             }
        
#         # Build combined search query
#         search_query = query
#         if environment:
#             search_query += f" {environment}"
#         if mood:
#             search_query += f" {mood}"
        
#         # Search in relevant fields
#         search_fields = [
#             'name', 'title', 'track_id', 'file_path', 'path',
#             'environment', 'location', 'setting',
#             'mood', 'style', 'group', 'category',
#             'description', 'tags', 'keywords'
#         ]
        
#         results = search_in_records(tracks, search_query, search_fields, max_results)
        
#         # Format results
#         formatted_results = []
#         for track in results:
#             track_id = (
#                 track.get('track_id') or 
#                 track.get('file_path') or 
#                 track.get('path') or 
#                 f"ambience/{track.get('name', 'unknown')}.mp3"
#             )
#             # Ensure track_id has ambience/ prefix
#             if not track_id.startswith('ambience/'):
#                 track_id = f"ambience/{track_id}"
            
#             formatted_results.append({
#                 "track_id": track_id,
#                 "name": track.get('name') or track.get('title', 'Unknown'),
#                 "environment": track.get('environment') or track.get('location', 'Unknown'),
#                 "mood": track.get('mood', 'Unknown'),
#                 "group": track.get('group') or track.get('category', 'Unknown'),
#                 "duration": track.get('duration', 'Unknown'),
#                 "description": track.get('description', '')
#             })
        
#         return {
#             "status": "success",
#             "tracks": formatted_results,
#             "query": query,
#             "total_found": len(formatted_results),
#             "message": f"Found {len(formatted_results)} ambience track(s) matching '{query}'"
#         }
        
#     except Exception as e:
#         error_msg = f"Ambience search failed: {str(e)}"
#         logger.error(f"[AmbienceSearch] {error_msg}")
#         return {
#             "status": "error",
#             "message": error_msg,
#             "tracks": [],
#             "query": query
#         }


# # ============================================================================
# # SFX SEARCH TOOL
# # ============================================================================

# def search_sfx(
#     query: str,
#     category: str = "",
#     max_results: int = 5,
# ) -> Dict[str, Any]:
#     """
#     Search for sound effects (SFX).
    
#     Use this tool to find sound effects related to actions like explosions,
#     gunshots, fighting, car crashes, as well as ambient sounds like footsteps,
#     crowd reactions, and bodily functions.
    
#     Args:
#         query: Search query describing the sound effect needed (e.g., "gunshot",
#                "door slam", "footsteps wood", "car engine", "explosion")
#         category: Filter for category - use "action", "nature", "vehicles",
#                  "human", "mechanical", etc. Leave empty for no filter.
#         max_results: Maximum number of results to return (default 5)
    
#     Returns:
#         A dictionary containing:
#         - status: "success" or "error"
#         - tracks: List of matching SFX tracks with track_id, name, category
#         - query: The original search query
#         - message: Description or error message
#     """
#     logger.info(f"[SFXSearch] Query: '{query}', category={category}")
    
#     try:
#         # Load knowledge base
#         tracks = load_knowledge_base(SFX_KB_PATH)
        
#         if not tracks:
#             return {
#                 "status": "error",
#                 "message": f"SFX knowledge base not found at {SFX_KB_PATH}. Please ensure the file exists.",
#                 "tracks": [],
#                 "query": query
#             }
        
#         # Build combined search query
#         search_query = query
#         if category:
#             search_query += f" {category}"
        
#         # Search in relevant fields
#         search_fields = [
#             'name', 'title', 'track_id', 'file_path', 'path',
#             'category', 'type', 'group',
#             'description', 'tags', 'keywords',
#             'action', 'sound'
#         ]
        
#         results = search_in_records(tracks, search_query, search_fields, max_results)
        
#         # Format results
#         formatted_results = []
#         for track in results:
#             track_id = (
#                 track.get('track_id') or 
#                 track.get('file_path') or 
#                 track.get('path') or 
#                 f"sfx/{track.get('name', 'unknown')}.mp3"
#             )
#             # Ensure track_id has sfx/ prefix
#             if not track_id.startswith('sfx/'):
#                 track_id = f"sfx/{track_id}"
            
#             formatted_results.append({
#                 "track_id": track_id,
#                 "name": track.get('name') or track.get('title', 'Unknown'),
#                 "category": track.get('category') or track.get('type', 'Unknown'),
#                 "duration": track.get('duration', 'Unknown'),
#                 "description": track.get('description', '')
#             })
        
#         return {
#             "status": "success",
#             "tracks": formatted_results,
#             "query": query,
#             "total_found": len(formatted_results),
#             "message": f"Found {len(formatted_results)} SFX track(s) matching '{query}'"
#         }
        
#     except Exception as e:
#         error_msg = f"SFX search failed: {str(e)}"
#         logger.error(f"[SFXSearch] {error_msg}")
#         return {
#             "status": "error",
#             "message": error_msg,
#             "tracks": [],
#             "query": query
#         }


# # ============================================================================
# # CREATE FUNCTION TOOLS
# # ============================================================================

# search_voices_tool = FunctionTool(search_voices)
# search_music_tool = FunctionTool(search_music)
# search_ambience_tool = FunctionTool(search_ambience)
# search_sfx_tool = FunctionTool(search_sfx)


# Copyright 2025
# Search Tools for Director Agent - Using Google File Search

"""
Search Tools - Search through knowledge base files using Google File Search API.

These tools use Google's File Search capability to perform semantic search
across uploaded knowledge base documents for:
- Voice profiles (ElevenLabs voices)
- Music tracks
- Ambience sounds
- Sound effects (SFX)

Based on the RAG agent example pattern.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from google import genai
from google.genai import types
from google.adk.tools import FunctionTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration ---
CONFIG_PATH = Path(__file__).parent.parent / "file_store_config.json"


def load_file_store_config() -> Dict[str, Any]:
    """Load the file search store configuration."""
    if not CONFIG_PATH.exists():
        return {
            'error': True,
            'message': 'No file store configured. Please run setup_file_stores.py first.'
        }
    
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def save_file_store_config(config: Dict[str, Any]):
    """Save the file search store configuration."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def get_genai_client():
    """Get the Google GenAI client."""
    api_key = os.getenv('FILE_SEARCH_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Neither FILE_SEARCH_API_KEY nor GOOGLE_API_KEY found in environment")
    return genai.Client(api_key=api_key)


def search_file_store(
    store_name: str,
    query: str,
    store_type: str = "general"
) -> Dict[str, Any]:
    """
    Search a file search store using Google's File Search API.
    
    Args:
        store_name: The name of the file search store
        query: The search query
        store_type: Type of store for logging purposes
        
    Returns:
        Dictionary with search results
    """
    try:
        client = get_genai_client()
        
        logger.info(f"[FileSearch:{store_type}] Searching store: {store_name}")
        logger.info(f"[FileSearch:{store_type}] Query: {query}")
        
        # Perform file search using Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )]
            )
        )
        
        # Extract grounding sources
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            if grounding and grounding.grounding_chunks:
                for c in grounding.grounding_chunks:
                    if c.retrieved_context:
                        sources.append(c.retrieved_context.title)
        logger.info(f"[FileSearch:{store_type}] Found {len(sources)} source(s)")
        
        return {
            "status": "success",
            "answer": response.text,
            "sources": list(set(sources)),
            "query": query
        }
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(f"[FileSearch:{store_type}] Error: {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "answer": "",
            "sources": [],
            "query": query
        }


# ============================================================================
# VOICE SEARCH TOOL
# ============================================================================

def search_voices(
    query: str,
    gender: str = "",
    age_range: str = "",
    tone: str = "",
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    Search for ElevenLabs voice profiles using semantic search.
    
    Use this tool to find voice profiles categorized by gender, age range,
    tone, and style, suitable for various character roles and narration.
    
    Args:
        query: Search query describing the voice needed (e.g., "young male soldier", 
               "elderly female narrator", "deep authoritative")
        gender: Filter for gender - use "male", "female", or "neutral". Leave empty for no filter.
        age_range: Filter for age - use "young", "middle-aged", or "elderly". Leave empty for no filter.
        tone: Filter for tone - use "warm", "authoritative", "soft", etc. Leave empty for no filter.
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - answer: The search results with voice recommendations
        - voices: Parsed voice profiles if available
        - query: The original search query
        - message: Description or error message
    """
    logger.info(f"[VoiceSearch] Query: '{query}', gender={gender}, age={age_range}, tone={tone}")
    
    # Load configuration
    config = load_file_store_config()
    
    if config.get('error'):
        return {
            "status": "error",
            "message": config['message'],
            "answer": "",
            "voices": [],
            "query": query
        }
    
    # Get the voices store name
    store_name = config.get('voices_store_name')
    if not store_name:
        return {
            "status": "error",
            "message": "Voices file store not configured. Please run setup_file_stores.py first.",
            "answer": "",
            "voices": [],
            "query": query
        }
    
    # Build enhanced search query
    search_query = f"Find voice profiles matching: {query}"
    if gender:
        search_query += f", gender: {gender}"
    if age_range:
        search_query += f", age: {age_range}"
    if tone:
        search_query += f", tone: {tone}"
    search_query += f". Return up to {max_results} matching voices with their voice_id, name, gender, age, tone, and description."
    
    # Perform search
    result = search_file_store(store_name, search_query, "voices")
    
    return {
        "status": result["status"],
        "answer": result.get("answer", ""),
        "voices": [],  # The answer contains the voice info in natural language
        "sources": result.get("sources", []),
        "query": query,
        "message": result.get("message", f"Search completed for '{query}'")
    }


# ============================================================================
# MUSIC SEARCH TOOL
# ============================================================================

def search_music(
    query: str,
    mood: str = "",
    genre: str = "",
    style: str = "",
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    Search for music tracks by mood, style, and genre using semantic search.
    
    Use this tool to find music tracks including orchestral, electronic,
    jazz, and piano pieces for various emotional contexts.
    
    Args:
        query: Search query describing the music needed (e.g., "tense action",
               "romantic piano", "dramatic orchestral")
        mood: Filter for mood - use "tense", "romantic", "sad", "uplifting", etc. Leave empty for no filter.
        genre: Filter for genre - use "orchestral", "electronic", "jazz", etc. Leave empty for no filter.
        style: Filter for style - use "cinematic", "ambient", "dramatic", etc. Leave empty for no filter.
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - answer: The search results with music recommendations
        - tracks: Parsed music tracks if available
        - query: The original search query
        - message: Description or error message
    """
    logger.info(f"[MusicSearch] Query: '{query}', mood={mood}, genre={genre}, style={style}")
    
    # Load configuration
    config = load_file_store_config()
    
    if config.get('error'):
        return {
            "status": "error",
            "message": config['message'],
            "answer": "",
            "tracks": [],
            "query": query
        }
    
    # Get the music store name
    store_name = config.get('music_store_name')
    if not store_name:
        return {
            "status": "error",
            "message": "Music file store not configured. Please run setup_file_stores.py first.",
            "answer": "",
            "tracks": [],
            "query": query
        }
    
    # Build enhanced search query
    search_query = f"Find music tracks matching: {query}"
    if mood:
        search_query += f", mood: {mood}"
    if genre:
        search_query += f", genre: {genre}"
    if style:
        search_query += f", style: {style}"
    search_query += f". Return up to {max_results} matching tracks with their track_id (file path), name, mood, genre, style, and duration."
    
    # Perform search
    result = search_file_store(store_name, search_query, "music")
    
    return {
        "status": result["status"],
        "answer": result.get("answer", ""),
        "tracks": [],  # The answer contains the track info in natural language
        "sources": result.get("sources", []),
        "query": query,
        "message": result.get("message", f"Search completed for '{query}'")
    }


# ============================================================================
# AMBIENCE SEARCH TOOL
# ============================================================================

def search_ambience(
    query: str,
    environment: str = "",
    mood: str = "",
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    Search for ambient sound effects using semantic search.
    
    Use this tool to find ambient sounds including natural environments,
    urban soundscapes, and crowd noises, categorized by group, style, and mood.
    
    Args:
        query: Search query describing the ambience needed (e.g., "forest birds",
               "city traffic", "hospital busy", "rain storm")
        environment: Filter for environment type - use "indoor", "outdoor",
                    "urban", "nature", "industrial", etc. Leave empty for no filter.
        mood: Filter for mood - use "peaceful", "tense", "busy", etc. Leave empty for no filter.
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - answer: The search results with ambience recommendations
        - tracks: Parsed ambience tracks if available
        - query: The original search query
        - message: Description or error message
    """
    logger.info(f"[AmbienceSearch] Query: '{query}', environment={environment}, mood={mood}")
    
    # Load configuration
    config = load_file_store_config()
    
    if config.get('error'):
        return {
            "status": "error",
            "message": config['message'],
            "answer": "",
            "tracks": [],
            "query": query
        }
    
    # Get the ambience store name
    store_name = config.get('ambience_store_name')
    if not store_name:
        return {
            "status": "error",
            "message": "Ambience file store not configured. Please run setup_file_stores.py first.",
            "answer": "",
            "tracks": [],
            "query": query
        }
    
    # Build enhanced search query
    search_query = f"Find ambient sound tracks matching: {query}"
    if environment:
        search_query += f", environment: {environment}"
    if mood:
        search_query += f", mood: {mood}"
    search_query += f". Return up to {max_results} matching tracks with their track_id (file path), name, environment, mood, and description."
    
    # Perform search
    result = search_file_store(store_name, search_query, "ambience")
    
    return {
        "status": result["status"],
        "answer": result.get("answer", ""),
        "tracks": [],  # The answer contains the track info in natural language
        "sources": result.get("sources", []),
        "query": query,
        "message": result.get("message", f"Search completed for '{query}'")
    }


# ============================================================================
# SFX SEARCH TOOL
# ============================================================================

def search_sfx(
    query: str,
    category: str = "",
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    Search for sound effects (SFX) using semantic search.
    
    Use this tool to find sound effects related to actions like explosions,
    gunshots, fighting, car crashes, as well as ambient sounds like footsteps,
    crowd reactions, and bodily functions.
    
    Args:
        query: Search query describing the sound effect needed (e.g., "gunshot",
               "door slam", "footsteps wood", "car engine", "explosion")
        category: Filter for category - use "action", "nature", "vehicles",
                 "human", "mechanical", etc. Leave empty for no filter.
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - answer: The search results with SFX recommendations
        - tracks: Parsed SFX tracks if available
        - query: The original search query
        - message: Description or error message
    """
    logger.info(f"[SFXSearch] Query: '{query}', category={category}")
    
    # Load configuration
    config = load_file_store_config()
    
    if config.get('error'):
        return {
            "status": "error",
            "message": config['message'],
            "answer": "",
            "tracks": [],
            "query": query
        }
    
    # Get the SFX store name
    store_name = config.get('sfx_store_name')
    if not store_name:
        return {
            "status": "error",
            "message": "SFX file store not configured. Please run setup_file_stores.py first.",
            "answer": "",
            "tracks": [],
            "query": query
        }
    
    # Build enhanced search query
    search_query = f"Find sound effect tracks matching: {query}"
    if category:
        search_query += f", category: {category}"
    search_query += f". Return up to {max_results} matching tracks with their track_id (file path), name, category, duration, and description."
    
    # Perform search
    result = search_file_store(store_name, search_query, "sfx")
    
    return {
        "status": result["status"],
        "answer": result.get("answer", ""),
        "tracks": [],  # The answer contains the track info in natural language
        "sources": result.get("sources", []),
        "query": query,
        "message": result.get("message", f"Search completed for '{query}'")
    }


# ============================================================================
# CREATE FUNCTION TOOLS
# ============================================================================

search_voices_tool = FunctionTool(search_voices)
search_music_tool = FunctionTool(search_music)
search_ambience_tool = FunctionTool(search_ambience)
search_sfx_tool = FunctionTool(search_sfx)