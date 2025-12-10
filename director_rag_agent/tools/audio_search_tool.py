"""
Audio Search Tools for Director Agent

Provides search_music, search_ambience, search_sfx functions
for the Audio Director agent to select appropriate audio tracks.
"""

from typing import Dict, Any, Optional
from director_rag_agent.tools.audio_search import (
    search_music as _search_music,
    search_ambience as _search_ambience,
    search_sfx as _search_sfx
)


def search_music(
    query: str,
    group: Optional[str] = None,
    style: Optional[str] = None,
    mood: Optional[str] = None,
    intensity_min: Optional[int] = None,
    intensity_max: Optional[int] = None,
    loopable: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Search for background music tracks using semantic search.
    
    Args:
        query: Semantic search string describing desired music 
               (e.g., "epic orchestral battle", "peaceful piano melody", "tense suspenseful")
        group: Filter by music category. Options:
               - "EPIC" - grand, heroic, powerful orchestral
               - "UPLIFTING" - hopeful, inspiring, bright
               - "TENSION" - suspenseful, anxious, building
               - "SAD" - melancholic, emotional, somber
               - "ACTION" - fast-paced, energetic, intense
               - "ROMANTIC" - love themes, tender, emotional
               - "HORROR" - scary, dark, unsettling
               - "COMEDY" - light, playful, humorous
               - "AMBIENT" - atmospheric, background, subtle
        style: Filter by musical style (e.g., "orchestral strings", "electronic", 
               "piano", "acoustic guitar", "synth")
        mood: Filter by mood (e.g., "epic", "peaceful", "tense", "hopeful", 
              "dark", "energetic", "melancholic")
        intensity_min: Minimum intensity level (0-100). Use for subtle background music.
        intensity_max: Maximum intensity level (0-100). Use to cap energy level.
        loopable: Filter for loopable tracks (True for background music that needs to loop)
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - items: List of matching music tracks with file paths
        - scores: Similarity scores
        - query: Original query
        
    Example results item:
        {
            "file": "music/m_orchestra_epic_03.mp3",
            "id": "m_orchestra_epic_03",
            "group": "EPIC",
            "style": "orchestra",
            "mood": ["epic", "heroic", "majestic"],
            "intensity": 85,
            "loopable": true,
            "title": "Orchestra Epic 03"
        }
    """
    intensity = None
    if intensity_min is not None or intensity_max is not None:
        intensity = (intensity_min or 0, intensity_max or 100)
    
    return _search_music(
        query=query,
        top_k=3,
        group=group,
        style=style,
        mood=mood,
        intensity=intensity,
        loopable=loopable
    )


def search_ambience(
    query: str,
    group: Optional[str] = None,
    style: Optional[str] = None,
    mood: Optional[str] = None,
    intensity_min: Optional[int] = None,
    intensity_max: Optional[int] = None,
    loopable: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Search for ambient background sounds using semantic search.
    
    Args:
        query: Semantic search string describing desired ambience
               (e.g., "busy city street", "quiet forest birds", "rain on window")
        group: Filter by ambience category. Options:
               - "People in the City" - urban crowds, traffic, street sounds
               - "Suburbs & Countryside" - rural, nature, quiet residential
               - "Interior Spaces" - rooms, offices, indoor environments
               - "Weather" - rain, wind, thunder, storms
               - "Nature" - forests, rivers, wildlife
               - "Industrial" - factories, machinery, construction
               - "Transportation" - cars, trains, airports
        style: Filter by style (e.g., "door sounds", "general effect", "crowd")
        mood: Filter by mood (e.g., "peaceful", "busy", "urban", "natural", "tense")
        intensity_min: Minimum intensity level (0-100)
        intensity_max: Maximum intensity level (0-100)
        loopable: Filter for loopable tracks (True for continuous background ambience)
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - items: List of matching ambience tracks with file paths
        - scores: Similarity scores
        - query: Original query
        
    Example results item:
        {
            "file": "ambience/a_countryside_birds_01.mp3",
            "id": "a_countryside_birds_01",
            "group": "Suburbs & Countryside",
            "style": "general effect",
            "mood": ["natural", "peaceful", "outdoors"],
            "intensity": 40,
            "loopable": true
        }
    """
    intensity = None
    if intensity_min is not None or intensity_max is not None:
        intensity = (intensity_min or 0, intensity_max or 100)
    
    return _search_ambience(
        query=query,
        top_k=3,
        group=group,
        style=style,
        mood=mood,
        intensity=intensity,
        loopable=loopable
    )


def search_sfx(
    query: str,
    group: Optional[str] = None,
    style: Optional[str] = None,
    mood: Optional[str] = None,
    intensity_min: Optional[int] = None,
    intensity_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Search for sound effects using semantic search.
    
    Args:
        query: Semantic search string describing desired sound effect
               (e.g., "gunshot explosion", "door slam", "footsteps running")
        group: Filter by SFX category. Options:
               - "Fighting" - punches, kicks, combat sounds
               - "Explosions" - blasts, booms, impacts
               - "Weapons" - guns, swords, projectiles
               - "Footsteps" - walking, running, various surfaces
               - "Doors" - opening, closing, knocking
               - "Vehicles" - cars, engines, crashes
               - "Technology" - beeps, computers, machines
               - "Body" - breathing, heartbeat, movement
               - "Glass" - breaking, shattering
               - "Metal" - clanging, scraping
        style: Filter by style (e.g., "general effect", "impact", "whoosh")
        mood: Filter by mood (e.g., "neutral", "intense", "subtle")
        intensity_min: Minimum intensity level (0-100)
        intensity_max: Maximum intensity level (0-100)
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - items: List of matching SFX tracks with file paths
        - scores: Similarity scores
        - query: Original query
        
    Example results item:
        {
            "file": "sfx/s_bar_fight_brawl_01.mp3",
            "id": "s_bar_fight_brawl_01",
            "group": "Fighting",
            "style": "general effect",
            "mood": ["neutral", "ambient", "cinematic"],
            "intensity": 40,
            "loopable": true
        }
    """
    GROUP_ALIASES = {
        "Human": "Body",
        "People": "Body", 
        "Voice": "Body",
        "Gunshots": "Weapons",
        "Guns": "Weapons",
    }
    
    if group and group in GROUP_ALIASES:
        group = GROUP_ALIASES[group]
        
    intensity = None
    if intensity_min is not None or intensity_max is not None:
        intensity = (intensity_min or 0, intensity_max or 100)
    
    return _search_sfx(
        query=query,
        top_k=3,
        group=group,
        style=style,
        mood=mood,
        intensity=intensity,
        loopable=None  # SFX typically don't need loopable filter
    )