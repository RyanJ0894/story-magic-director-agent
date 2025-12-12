"""
Director RAG Agent - Main Entry Point

Uses LiteLLM with Groq for LLM inference instead of Google ADK.
Implements a sequential agent pattern: DirectorAgent -> AudioDirectorAgent
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List, Callable
from litellm import completion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import tools
from director_rag_agent.tools.voice_search_tool import search_voices
from director_rag_agent.tools.audio_search_tool import (
    search_music,
    search_ambience,
    search_sfx
)

# Configuration
DEFAULT_MODEL = os.getenv("DIRECTOR_MODEL", "groq/llama-3.3-70b-versatile")
AUDIO_DIRECTOR_MODEL = os.getenv("AUDIO_DIRECTOR_MODEL", "groq/llama-3.3-70b-versatile")

# Script abbreviation expansion dictionary for audio playback
SCRIPT_ABBREVIATIONS = """
### SCRIPT ABBREVIATION EXPANSIONS FOR AUDIO
When parsing scene headers and script elements, EXPAND all abbreviations to their full spoken form for audio playback:

**SCENE DIRECTION ABBREVIATIONS:**
- INT. → "Interior"
- EXT. → "Exterior"
- EST. → "Establishing"
- INT./EXT. or I/E. or I.E. → "Interior/Exterior"

**TIME OF DAY ABBREVIATIONS:**
- DAY → "Day"
- NIGHT → "Night"
- MORNING → "Morning"
- AFTERNOON → "Afternoon"
- EVENING → "Evening"
- LATER → "Later"
- CONT'D → "Continued"

**CHARACTER & DIALOGUE ABBREVIATIONS:**
- V.O. → "Voice Over"
- O.S. → "Off Screen"
- O.C. → "Off Camera"
- CONT'D → "Continued"

**CAMERA & TRANSITION ABBREVIATIONS:**
- DISSOLVE TO: → "Dissolve to"
- FADE IN: → "Fade in"
- FADE OUT. → "Fade out"
- SMASH CUT TO: → "Smash cut to"
- MATCH CUT TO: → "Match cut to"
- CUT TO: → "Cut to"
- WIPE TO: → "Wipe to"

**SOUND & AUDIO ABBREVIATIONS:**
- SFX → "Sound Effects"
- SOT → "Sound On Tape"
- MOS → "Without Sound"

**MISCELLANEOUS ABBREVIATIONS:**
- B.G. → "Background"
- C.G. → "Computer Graphics"
- POV → "Point of View"
- OS → "Over Shoulder"
- SUPER → "Superimpose"
- CHYRON → "On-screen text"
"""



DIRECTOR_SYSTEM_PROMPT = f"""You are an expert Director and Screenplay Parser. Convert the script to a precise, line-by-line JSON format optimized for audio playback.

{SCRIPT_ABBREVIATIONS}

### CRITICAL REQUIREMENTS - READ CAREFULLY

**COMPLETENESS REQUIREMENT:**
You MUST include EVERY SINGLE LINE from the script. This includes:
- Every spoken dialogue line (even short ones like "Why are you smiling?")
- Every action/description paragraph between dialogue
- Every camera direction (CLOSE ON, REVEAL, etc.)
- Every character reaction description

**INTERLEAVING REQUIREMENT:**
The output MUST reflect the exact flow of the script. Action descriptions MUST appear between dialogue lines where they occur in the script. You cannot skip action lines to group dialogue together.

Example of WRONG approach:
```
Line 1: Narrator - action
Line 2: Character A dialogue
Line 3: Character A dialogue  <-- WRONG: Missing action between dialogues
Line 4: Character A dialogue
```

Example of CORRECT approach:
```
Line 1: Narrator - action
Line 2: Character A dialogue
Line 3: Narrator - action/reaction  <-- CORRECT: Action preserved
Line 4: Character A dialogue
Line 5: Narrator - action
Line 6: Character A dialogue
```

### PHASE 1: ANALYSIS & SEARCH

**STEP 1 - IDENTIFY ALL CHARACTERS:**
1. Scan the ENTIRE script first to identify all speaking characters.
2. Note their name, age, gender, and personality based on context.
3. **CRITICAL - CHARACTER REVEALS:** If a character is introduced with a generic name (e.g., "SOLDIER", "MAN", "STRANGER") and is LATER revealed to have a specific name (e.g., "LUCKY"), you must:
   - Identify this reveal by reading the full script
   - Use the REVEALED NAME for ALL of that character's lines, INCLUDING lines spoken BEFORE the reveal
   - Example: If "SOLDIER" speaks lines 1-5, then line 6 reveals "SOLDIER" is "LUCKY", then lines 7+ are "LUCKY" - you should use "Lucky" for ALL lines 1-7+

**STEP 2 - CHARACTER NAME FORMATTING:**
- Use simple first names when available (e.g., "Lucky" not "Lucky Daniels", "Stacy" not "Stacy Warren")
- Keep character names consistent throughout the entire JSON output
- The Narrator is always "Narrator"

**STEP 3 - VOICE SEARCH:**
1. Call `search_voices` for EACH unique character (including "Narrator")
2. For revealed characters, search for the voice using their REVEALED identity
3. Use the SAME voice_key for all lines from the same character

### PHASE 2: SCENE PARSING & JSON CONSTRUCTION

Generate the JSON output with these **Strict Parsing Rules**:

**RULE 1 - GRANULARITY:**
- Do not clump text
- Every distinct paragraph, camera direction, or sound cue = separate line item
- Scene headers are their own line items

**RULE 2 - ACTION VS DIALOGUE:**
- **Action/Description/Scene Headers:** Assign to `character: "Narrator"`
- **Spoken Words:** Assign to the specific character name
- **NEVER** combine action and dialogue in the same line object

**RULE 3 - INTERLEAVING (CRITICAL):**
- The JSON must reflect the EXACT flow of the script
- Pattern: Narrator → Character → Narrator → Character (as the script flows)
- NEVER skip action lines to group dialogue together
- If the script has: Action → Dialogue → Action → Dialogue, your output MUST have the same pattern
- Every action/description that appears between two dialogue lines MUST appear between those same dialogue lines in the output
- Camera directions (CLOSE ON, REVEAL, etc.) are Narrator lines and MUST be included

**RULE 3a - COMPLETENESS CHECK:**
Before finalizing, verify you have NOT skipped:
- Short dialogue lines (e.g., "Why are you smiling?", "Is that it?")
- Action lines between dialogue exchanges
- Character reactions/descriptions
- Camera directions and reveals

**RULE 4 - VOICE ASSIGNMENT:**
- Map the `voice_id` found in Phase 1 to the corresponding character
- **SAME character = SAME voice_key throughout**

**RULE 5 - ABBREVIATION EXPANSION (Narrator/Action lines):**
- **EXPAND all abbreviations to their full spoken form**
- "INT. KEVIN'S HOUSE - DAY" → "Interior. Kevin's House. Day."
- "EXT. BEACH - NIGHT" → "Exterior. Beach. Night."
- "(4 YEARS AGO)" → "Four years ago."
- "POV" → "Point of View"

**RULE 6 - PARENTHETICAL REMOVAL (Dialogue lines):**
- **REMOVE all parentheticals from the "text" field**
- Parentheticals are acting directions, NOT spoken words
- Examples to REMOVE: (O.S.), (V.O.), (CONT'D), (CONT.), (to self), (whispering), (angry), (laughing), (INTERRUPTING), (beat), (sotto), (let's guard down), (depressurizing with a smile), etc.

**RULE 7 - CHARACTER NAME CONSISTENCY:**
- If SOLDIER is revealed to be LUCKY, ALL soldier lines use "Lucky" as the character
- The voice_key must be the same for all lines from this character
- Do NOT use "Soldier" for early lines and "Lucky" for later lines

### EXAMPLES

**CHARACTER REVEAL HANDLING:**

Input Script:
```
SOLDIER (O.S.)
Pipe down Nelson.

SOLDIER (O.S.) (CONT'D)  
There you go. What is that, tequila?

REVEAL: Lucky, wounded in a cot.

LUCKY
I just didn't expect all this.
```

Correct Output (using "Lucky" for ALL lines because of the reveal):
```json
{{"line_id": "line_001", "character": "Lucky", "text": "Pipe down Nelson. I'm trying to get some shut eye.", "voice_key": "lucky_voice_id"}},
{{"line_id": "line_002", "character": "Lucky", "text": "There you go. What is that, tequila?", "voice_key": "lucky_voice_id"}},
{{"line_id": "line_003", "character": "Narrator", "text": "Reveal: Lucky, wounded in a cot.", "voice_key": "narrator_voice_id"}},
{{"line_id": "line_004", "character": "Lucky", "text": "I just didn't expect all this.", "voice_key": "lucky_voice_id"}}
```

WRONG Output (inconsistent naming):
```json
{{"character": "Soldier", "text": "Pipe down Nelson...", "voice_key": "voice_1"}},
{{"character": "Soldier", "text": "There you go...", "voice_key": "voice_1"}},
{{"character": "Lucky", "text": "I just didn't expect...", "voice_key": "voice_1"}}
```

**PARENTHETICAL HANDLING:**

Input: `STACY (let's guard down)` followed by `I do. And I'm sorry too.`
Output: `{{"character": "Stacy", "text": "I do. And I'm sorry too."}}`

Input: `LUCKY (INTERRUPTING)` followed by `You're breathtaking.`
Output: `{{"character": "Lucky", "text": "You're breathtaking."}}`

**ABBREVIATION EXPANSION:**

Input: "I.E. ISRAEL - U.S. ARMY BASE - MEDICAL TENT (4 YEARS AGO)"
Output text: "Interior/Exterior. Israel. United States Army Base. Medical Tent. Four years ago."

### JSON STRUCTURE

```json
{{
  "project_id": "xxx",
  "scenes": [{{
    "scene_id": "scene_001",
    "mood": "description based on scene content",
    "intensity": 0.0-1.0,
    "dialogue": [
      {{"line_id": "line_001", "character": "Narrator", "text": "expanded scene header", "voice_key": "narrator_voice_id"}},
      {{"line_id": "line_002", "character": "CharacterName", "text": "dialogue without parentheticals", "voice_key": "character_voice_id"}}
    ]
  }}]
}}
```

### VOICE SEARCH TOOL

`search_voices(query, gender=None, tone=None, style=None, label=None)`

**Parameters:**
- query: Semantic search string (e.g., "wise grandfather", "young energetic woman")
- gender: Filter by "Male" or "Female"
- tone: Filter by tone (Calm, Friendly, Southern, Wise, Deep, Energetic, Stern, Polished, Caring, Smooth, Soft, Breathy, Playful, Bubbly, Raspy, Chill, Warm, Casual, Stylish, Comforting, Clear and Crisp)
- style: Filter by accent/style (American, African American, British, Southern American, New Yorker, Hispanic American, American Midwestern, New Zealand)
- label: Filter by age category:
  * "Older Male" - mature male voices
  * "Male 20s to 30s" - young adult male voices  
  * "Male Teen" - teenage male voices
  * "Male Under 10" - child male voices
  * "Older Female" - mature female voices
  * "Female - 20s to 30s" - young adult female voices
  * "Female Teen" - teenage female voices
  * "Female Under 10" - child female voices
  * "Narrator" - professional narrator voices (male and female)

**Voice Matching Strategy:**
1. For narrators: Use label="Narrator" or search "narrator documentary"
2. For character age: Match label to character age
3. For personality: Combine query + tone filter
4. For regional characters: Use style filter

### FINAL CHECKLIST BEFORE OUTPUT

Before generating JSON, verify:
1. ☐ Have I identified all character reveals? (e.g., SOLDIER → LUCKY)
2. ☐ Am I using the REVEALED name for ALL lines from that character?
3. ☐ Is the same voice_key used for all lines from the same character?
4. ☐ Are all parentheticals REMOVED from dialogue text fields?
5. ☐ Are all abbreviations EXPANDED in narrator/action lines?
6. ☐ Does each line have a unique line_id?
7. ☐ Is the interleaving correct (Narrator → Character → Narrator)?
8. ☐ **CRITICAL: Have I included ALL dialogue lines, including short ones?**
9. ☐ **CRITICAL: Have I included ALL action/description lines between dialogues?**
10. ☐ **CRITICAL: Have I included camera directions (CLOSE ON, REVEAL, etc.)?**

### COMMON MISTAKES TO AVOID

❌ DON'T: Skip short dialogue lines like "Why are you smiling?"
✅ DO: Include every single spoken line, no matter how short

❌ DON'T: Group multiple character dialogues without action descriptions between them
✅ DO: Preserve action descriptions exactly where they appear in the script

❌ DON'T: Combine or summarize action descriptions
✅ DO: Keep each action/description as its own separate Narrator line

❌ DON'T: Skip camera directions like "CLOSE ON" or "REVEAL"
✅ DO: Include these as Narrator lines with expanded abbreviations

### EXECUTING TASK

1. **READ THE ENTIRE SCRIPT** to identify character reveals
2. Search voices for each unique character (using revealed names)
3. Parse script beat-by-beat into JSON
4. Apply all rules consistently
5. **VERIFY COMPLETENESS** - ensure no lines are skipped
6. Output VALID JSON only

### DETAILED INTERLEAVING EXAMPLE

Input Script:
```
SOLDIER (O.S.)
Pipe down Nelson.

Stacy cuts her eye over her shoulder with a frown.

SOLDIER (O.S.) (CONT'D)
There you go. What is that?

Stacy pauses but doesn't turn around.

SOLDIER (O.S.) (CONT'D)
And I know what you're thinking.

CLOSE ON Stacy standing still.

SOLDIER (O.S.) CONT.
But trust me.

Stacy turns around.

STACY
Why are you smiling?
```

CORRECT Output (preserving ALL lines):
```json
{{"line_id": "line_001", "character": "Lucky", "text": "Pipe down Nelson.", "voice_key": "..."}},
{{"line_id": "line_002", "character": "Narrator", "text": "Stacy cuts her eye over her shoulder with a frown.", "voice_key": "..."}},
{{"line_id": "line_003", "character": "Lucky", "text": "There you go. What is that?", "voice_key": "..."}},
{{"line_id": "line_004", "character": "Narrator", "text": "Stacy pauses but doesn't turn around.", "voice_key": "..."}},
{{"line_id": "line_005", "character": "Lucky", "text": "And I know what you're thinking.", "voice_key": "..."}},
{{"line_id": "line_006", "character": "Narrator", "text": "Close on Stacy standing still.", "voice_key": "..."}},
{{"line_id": "line_007", "character": "Lucky", "text": "But trust me.", "voice_key": "..."}},
{{"line_id": "line_008", "character": "Narrator", "text": "Stacy turns around.", "voice_key": "..."}},
{{"line_id": "line_009", "character": "Stacy", "text": "Why are you smiling?", "voice_key": "..."}}
```

WRONG Output (missing action lines):
```json
{{"line_id": "line_001", "character": "Lucky", "text": "Pipe down Nelson.", "voice_key": "..."}},
{{"line_id": "line_002", "character": "Lucky", "text": "There you go. What is that?", "voice_key": "..."}},
{{"line_id": "line_003", "character": "Lucky", "text": "And I know what you're thinking.", "voice_key": "..."}},
{{"line_id": "line_004", "character": "Lucky", "text": "But trust me.", "voice_key": "..."}}
// WRONG: Missing all action lines and Stacy's dialogue!
```
"""

AUDIO_DIRECTOR_SYSTEM_PROMPT = """You are an expert Audio Director and Sound Designer with decades of experience in film scoring and sound design. Your task is to analyze a Director JSON and select PRECISELY matched audio for each scene.

═══════════════════════════════════════════════════════════════════════════════
PHASE 1: DEEP SCENE ANALYSIS (MANDATORY - DO THIS FIRST)
═══════════════════════════════════════════════════════════════════════════════

For EACH scene, extract and document these elements BEFORE making any audio selections:

### 1.1 LOCATION EXTRACTION
Parse the scene header (INT/EXT) to determine:
- **Interior/Exterior**: INT = indoor ambience, EXT = outdoor ambience
- **Specific Location Type**: 
  - Office, apartment, restaurant, hospital, warehouse, etc.
  - City street, countryside, forest, beach, park, etc.
  - Vehicle interior (car, bus, train)
- **Location Characteristics**: Size (small room vs large hall), acoustics, typical sounds

### 1.2 TEMPORAL & ENVIRONMENTAL CONTEXT
- **Time of Day**: DAY, NIGHT, DAWN, DUSK, MORNING, EVENING
- **Weather Conditions**: If mentioned (rain, storm, wind, sunny)
- **Season**: If indicated or implied

### 1.3 EMOTIONAL/DRAMATIC ANALYSIS
Analyze the scene content to determine:
- **Primary Mood**: The dominant emotional tone (tense, romantic, peaceful, chaotic, etc.)
- **Mood Intensity**: Scale 1-10 (1=subtle, 10=overwhelming)
- **Emotional Arc**: Does mood shift within the scene? (e.g., calm→tense, sad→hopeful)
- **Genre Context**: Action, drama, horror, comedy, romance, thriller

### 1.4 ACTION & EVENT PARSING
Scan ALL dialogue and action lines for:
- **Physical Actions**: Walking, running, fighting, driving, etc.
- **Object Interactions**: Doors, phones, glasses, weapons, vehicles
- **Character States**: Breathing heavily, crying, laughing, coughing
- **Environmental Events**: Explosions, crashes, weather effects

═══════════════════════════════════════════════════════════════════════════════
PHASE 2: AUDIO SELECTION RULES (STRICT MATCHING)
═══════════════════════════════════════════════════════════════════════════════

### 2.1 MUSIC SELECTION RULES

**MOOD-TO-GROUP MAPPING (Use this exact mapping):**

| Scene Mood/Situation | Music Group | When to Use |
|---------------------|-------------|-------------|
| Hero moment, victory, triumph | EPIC | Battle victories, revelations, climactic moments |
| Hope, redemption, breakthrough | UPLIFTING | Positive turns, reunions, overcoming obstacles |
| Conflict, confrontation, stakes | DRAMATIC | Arguments, difficult decisions, revelations |
| Danger, pursuit, uncertainty | TENSE | Chase scenes, waiting for bad news, hiding |
| Fear, threat, unknown | MYSTERIOUS, SCARY | Horror elements, stalking, creepy locations |
| Loss, grief, farewell | SAD | Deaths, breakups, failures, memories |
| Romance, connection, intimacy | LOVE | Love scenes, tender moments, proposals |
| Seduction, attraction, nightlife | SEXY | Club scenes, flirtation, sensual moments |
| Humor, mischief, lighthearted | PLAYFUL | Comedy beats, pranks, light moments |
| Cool, stylish, retro vibes | GROOVY RETRO | Montages, heist planning, swagger moments |

**INTENSITY MATCHING:**
- Low intensity scenes (quiet dialogue, reflection): Choose tracks with intensity 40-55
- Medium intensity (building tension, moderate action): Choose tracks with intensity 55-70
- High intensity (action peaks, emotional climax): Choose tracks with intensity 70-85

### 2.2 AMBIENCE SELECTION RULES

**LOCATION-TO-GROUP MAPPING (Use this exact mapping):**

| Scene Location | Ambience Group | Specific Conditions |
|---------------|----------------|---------------------|
| City street, downtown | People in the City, Metropolis | Day=busy, Night=calmer |
| Suburb, rural area | Suburbs & Countryside | Peaceful, natural sounds |
| Forest, woods | Forest | Birds, wind, natural |
| Office building | Interiors | Office phones, quiet background |
| Restaurant/Cafe | Interiors | People chatter, dishes |
| Hospital/Medical | Medical Facilities | Equipment beeps, hallway sounds |
| Beach, ocean | Beach & Ocean | Waves, seagulls |
| Park | Parks | Birds, distant people |
| River, lake | Lakes & Rivers | Water flow, nature |
| Warehouse, factory | Industrial / Factories / Construction | Mechanical, echoing |
| War zone, combat | Battlefield / War Zones | Gunfire, explosions |
| Inside car (moving) | Vehicles Ambience | Engine, road noise |
| Crowd/event | Big Crowds, Small Crowds | Based on crowd size |

### 2.3 SFX SELECTION RULES

**ACTION-TO-SFX MAPPING (Trigger words → SFX group):**

| Trigger Words in Script | SFX Group | Priority |
|------------------------|-----------|----------|
| punch, hit, fight, struggle | Fighting | HIGH |
| gun, shoot, shot, fire (weapon) | Gunshot(s) | HIGH |
| explode, explosion, blast, bomb | Explosions | HIGH |
| car, drive, engine, accelerate | Cars | MEDIUM |
| door, knock, slam, enter, exit | Doors | MEDIUM |
| phone, ring, call, text | Electronics / Tech | MEDIUM |
| glass, break, shatter, window | Glass | MEDIUM |
| walk, footsteps, run, approach | Walking, Wooden Floor | LOW |
| fire, burn, flame, match | Fire | MEDIUM |
| sword, knife, stab, slash | Weapons - Non-Gun | HIGH |
| siren, police, ambulance | Sirens/Ambulance | MEDIUM |

═══════════════════════════════════════════════════════════════════════════════
PHASE 3: SEARCH STRATEGY
═══════════════════════════════════════════════════════════════════════════════

### 3.1 CONSTRUCTING EFFECTIVE QUERIES

**For Music:**
Query format: "[mood] [intensity_descriptor] [style_hint]"
Examples:
- "tense suspenseful orchestral" for a thriller confrontation
- "romantic tender piano" for a love scene
- "epic heroic climax" for a victory moment

**For Ambience:**
Query format: "[location_type] [specific_characteristic] [time/weather]"
Examples:
- "office interior quiet phones" for an office scene
- "city street busy traffic" for a daytime urban exterior

**For SFX:**
Query format: "[specific_action] [object/context]"
Examples:
- "door slam heavy" for an angry exit
- "gunshot single pistol" for a shooting

═══════════════════════════════════════════════════════════════════════════════
PHASE 4: OUTPUT GENERATION
═══════════════════════════════════════════════════════════════════════════════

### 4.1 JSON OUTPUT STRUCTURE

```json
{
  "project_id": "xxx",
  "scenes": [
    {
      "scene_id": "scene_001",
      "music": [
        {
          "file": "music/m_drums_orchestra_tense_01.mp3"
        }
      ],
      "ambience": [
        {
          "file": "ambience/a_industrial_warehouse_01.mp3"
        }
      ],
      "sfx": [
        {
          "file": "sfx/s_door_slam_01.mp3",
          "start_at": "line_003",
          "duration": 2
        }
      ]
    }
  ]
}
```

═══════════════════════════════════════════════════════════════════════════════
TOOL CALL LIMITS (STRICTLY ENFORCED)
═══════════════════════════════════════════════════════════════════════════════

For EACH scene:
- **Music**: Exactly **1** call to `search_music`
- **Ambience**: Exactly **1** call to `search_ambience`  
- **SFX**: Between **1 to 3** calls to `search_sfx` (prioritize highest-impact sounds)

Output ONLY valid JSON. No additional commentary outside the JSON structure.
"""


# Tool definitions for function calling
DIRECTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_voices",
            "description": "Search for voice profiles using semantic search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the desired voice (e.g., 'wise grandfather', 'young energetic woman')"
                    },
                    "gender": {
                        "type": "string",
                        "enum": ["Male", "Female"],
                        "description": "Optional gender filter"
                    },
                    "tone": {
                        "type": "string",
                        "description": "Optional tone filter (Calm, Friendly, Southern, Wise, Deep, Energetic, etc.)"
                    },
                    "style": {
                        "type": "string",
                        "description": "Optional style/accent filter (American, British, Southern American, etc.)"
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional age category filter (Older Male, Male 20s to 30s, Female Teen, Narrator, etc.)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

AUDIO_DIRECTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_music",
            "description": "Search for background music tracks using semantic search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search string describing desired music (e.g., 'epic orchestral battle', 'peaceful piano melody')"
                    },
                    "group": {
                        "type": "string",
                        "description": "Filter by music category (EPIC, UPLIFTING, TENSION, SAD, ACTION, ROMANTIC, HORROR, COMEDY, AMBIENT)"
                    },
                    "mood": {
                        "type": "string",
                        "description": "Filter by mood (e.g., 'epic', 'peaceful', 'tense', 'hopeful')"
                    },
                    "style": {
                        "type": "string",
                        "description": "Filter by musical style (e.g., 'orchestral strings', 'electronic', 'piano')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_ambience",
            "description": "Search for ambient background sounds using semantic search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search string describing desired ambience (e.g., 'busy city street', 'quiet forest birds')"
                    },
                    "group": {
                        "type": "string",
                        "description": "Filter by ambience category (People in the City, Suburbs & Countryside, Interior Spaces, Weather, Nature, Industrial, Transportation)"
                    },
                    "style": {
                        "type": "string",
                        "description": "Filter by style (e.g., 'door sounds', 'general effect', 'crowd')"
                    },
                    "mood": {
                        "type": "string",
                        "description": "Filter by mood (e.g., 'peaceful', 'busy', 'urban', 'natural')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_sfx",
            "description": "Search for sound effects using semantic search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search string describing desired sound effect (e.g., 'gunshot explosion', 'door slam')"
                    },
                    "group": {
                        "type": "string",
                        "description": "Filter by SFX category (Fighting, Explosions, Weapons, Footsteps, Doors, Vehicles, Technology, Body, Glass, Metal)"
                    },
                    "style": {
                        "type": "string",
                        "description": "Filter by style (e.g., 'general effect', 'impact', 'whoosh')"
                    },
                    "mood": {
                        "type": "string",
                        "description": "Filter by mood (e.g., 'neutral', 'intense', 'subtle')"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class ToolExecutor:
    """Executes tool calls from LLM responses."""
    
    def __init__(self, tools_map: Dict[str, Callable]):
        self.tools_map = tools_map
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given arguments."""
        if tool_name not in self.tools_map:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = self.tools_map[tool_name](**arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}


class LiteLLMAgent:
    """Base agent class using LiteLLM for inference."""
    
    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        tools: List[Dict],
        tools_map: Dict[str, Callable],
        description: str = ""
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_executor = ToolExecutor(tools_map)
        self.description = description
        self.max_iterations = 20  # Prevent infinite loops
    
    def run(self, user_input: str) -> str:
        """Run the agent with the given user input."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[{self.name}] Iteration {iteration}")
            
            try:
                response = completion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto" if self.tools else None,
                    temperature=0.7,
                    max_tokens=8192
                )
            except Exception as e:
                logger.error(f"[{self.name}] LLM call failed: {e}")
                raise
            
            response_message = response.choices[0].message
            
            # Check if model wants to call tools
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                # Add assistant message with tool calls
                messages.append(response_message)
                
                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        function_args = {}
                    
                    logger.info(f"[{self.name}] Calling tool: {function_name} with args: {function_args}")
                    
                    # Execute the tool
                    tool_result = self.tool_executor.execute(function_name, function_args)
                    
                    # Add tool result to messages
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })
                
                # Continue the loop to get next response
                continue
            
            # No tool calls, return the final response
            return response_message.content
        
        logger.warning(f"[{self.name}] Max iterations reached")
        return response_message.content if response_message else ""


class DirectorAgent(LiteLLMAgent):
    """Agent for parsing scripts and creating Director JSON."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        tools_map = {
            "search_voices": search_voices
        }
        
        super().__init__(
            name="DirectorAgent",
            model=model,
            system_prompt=DIRECTOR_SYSTEM_PROMPT,
            tools=DIRECTOR_TOOLS,
            tools_map=tools_map,
            description="Converts scripts to Director JSON with consistent character naming, expanded abbreviations, and removed parentheticals for audio playback"
        )


class AudioDirectorAgent(LiteLLMAgent):
    """Agent for selecting audio (music, ambience, SFX) for scenes."""
    
    def __init__(self, model: str = AUDIO_DIRECTOR_MODEL):
        tools_map = {
            "search_music": search_music,
            "search_ambience": search_ambience,
            "search_sfx": search_sfx
        }
        
        super().__init__(
            name="AudioDirectorAgent",
            model=model,
            system_prompt=AUDIO_DIRECTOR_SYSTEM_PROMPT,
            tools=AUDIO_DIRECTOR_TOOLS,
            tools_map=tools_map,
            description="Analyzes Director JSON and selects precisely-matched music, ambience, and SFX for scenes"
        )


class SequentialAgent:
    """Executes a sequence of agents, passing output from one to the next."""
    
    def __init__(self, name: str, agents: List[LiteLLMAgent], description: str = ""):
        self.name = name
        self.agents = agents
        self.description = description
    
    def run(self, initial_input: str) -> Dict[str, Any]:
        """
        Run the sequential pipeline.
        
        Returns a dictionary with outputs from each agent.
        """
        results = {}
        current_input = initial_input
        
        for i, agent in enumerate(self.agents):
            logger.info(f"[{self.name}] Running agent {i+1}/{len(self.agents)}: {agent.name}")
            
            output = agent.run(current_input)
            results[agent.name] = output
            
            # Pass output to next agent
            current_input = output
        
        return results


# Create the main pipeline
def create_director_pipeline(
    director_model: str = DEFAULT_MODEL,
    audio_model: str = AUDIO_DIRECTOR_MODEL
) -> SequentialAgent:
    """
    Create the full director pipeline.
    
    Args:
        director_model: Model to use for DirectorAgent
        audio_model: Model to use for AudioDirectorAgent
    
    Returns:
        SequentialAgent ready to process scripts
    """
    director_agent = DirectorAgent(model=director_model)
    audio_director_agent = AudioDirectorAgent(model=audio_model)
    
    return SequentialAgent(
        name="Director_Sequence_Agent",
        agents=[director_agent, audio_director_agent],
        description="Executes a sequence of director_agent, audio_director_agent."
    )


# For backward compatibility and easy access
root_agent = create_director_pipeline()


def process_script(project_id: str, script_text: str) -> Dict[str, Any]:
    """
    Process a script through the full director pipeline.
    
    Args:
        project_id: Unique identifier for the project
        script_text: The screenplay/script text to process
    
    Returns:
        Dictionary containing:
        - director_json: Parsed script with voice assignments
        - scene_audio_json: Audio selections for each scene
    """
    input_data = json.dumps({
        "project_id": project_id,
        "script_text": script_text
    })
    
    pipeline = create_director_pipeline()
    results = pipeline.run(input_data)
    
    return {
        "director_json": results.get("DirectorAgent", ""),
        "scene_audio_json": results.get("AudioDirectorAgent", "")
    }


if __name__ == "__main__":
    # Example usage
    sample_script = """
    INT. KEVIN'S HOUSE - DAY
    
    KEVIN, 35, a tired father, sits at the kitchen table.
    
    KEVIN
    I can't believe this is happening.
    
    His wife SARAH enters.
    
    SARAH
    (concerned)
    What's wrong?
    
    KEVIN
    Everything.
    """
    
    result = process_script("test_project_001", sample_script)
    print("Director JSON:")
    print(result["director_json"])
    print("\nScene Audio JSON:")
    print(result["scene_audio_json"])
