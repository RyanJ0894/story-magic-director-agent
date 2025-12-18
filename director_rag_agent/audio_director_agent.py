"""
Audio Director Agent Module

Standalone module for the AudioDirectorAgent using LiteLLM with Groq.
"""

import os
import logging
import json
from typing import Dict, Any, List, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("AUDIO_DIRECTOR_MODEL", "groq/llama-3.3-70b-versatile")

# Load audio selection knowledge base
KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "knowledge", "audio_selection_knowledge.json")
AUDIO_KNOWLEDGE = {}

try:
    with open(KNOWLEDGE_FILE, 'r') as f:
        AUDIO_KNOWLEDGE = json.load(f)
    logger.info(f"Loaded audio knowledge: {len(AUDIO_KNOWLEDGE.get('voice_selection_examples', []))} voice examples, {len(AUDIO_KNOWLEDGE.get('music_examples', []))} music examples")
except FileNotFoundError:
    logger.warning(f"Audio knowledge file not found at {KNOWLEDGE_FILE}")
except Exception as e:
    logger.error(f"Error loading audio knowledge: {e}")
  
from litellm import completion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("AUDIO_DIRECTOR_MODEL", "groq/llama-3.3-70b-versatile")

# Build system prompt with knowledge base
KNOWLEDGE_SECTION = ""

if AUDIO_KNOWLEDGE:
   KNOWLEDGE_SECTION = """
═══════════════════════════════════════════════════════════════════════════════
MANDATORY KNOWLEDGE BASE - YOU MUST USE THESE EXACT PATTERNS
═══════════════════════════════════════════════════════════════════════════════

CRITICAL INSTRUCTION: The examples below are NOT suggestions - they are REQUIREMENTS.
When you see a character/scene that matches an example, you MUST use the EXACT voice_id/file shown.

**VOICE SELECTION EXAMPLES (USE THESE EXACT voice_ids):**

**VOICE SELECTION EXAMPLES:**
"""
    # Add voice examples
    for ex in AUDIO_KNOWLEDGE.get('voice_selection_examples', [])[:15]:
        KNOWLEDGE_SECTION += f"- {ex['character_name']} ({ex['character_description']}): voice_id={ex['correct_voice_id']}, label={ex['label']}, style={ex['style']}, tone={ex['tone']}\n"
    
 # CRITICAL VOICE RULES
    KNOWLEDGE_SECTION += """

CRITICAL VOICE RULES - VIOLATIONS ARE ERRORS:
- If character age is 17, 18, or 19 → MUST use "Male Teen" or "Female Teen" label
- If character is African American → MUST use African American style voice
- If you see "Derek, 18, African American" → MUST use voice_id: DTKMou8ccj1ZaWGBiotd
- If you see "Drew, 17, Caucasian" → MUST use voice_id: Pb8RZcHs3ga4StE7wiPM
- If you see "Frankie, 17" → MUST use voice_id: 3XOBzXhnDY98yeWQ3GdM

INCORRECT EXAMPLES (DO NOT DO THIS):
- Using "Male 20s to 30s" for a 17-year-old = WRONG
- Using "American" style for African American character = WRONG

"""   

   KNOWLEDGE_SECTION += """
**VOICE SELECTION RULES:**
- Ages 11-19 (including 17, 18, 19) = Male Teen or Female Teen labels (NOT 20s to 30s!)
- Ages 20-39 = Male 20s to 30s or Female 20s to 30s labels
- Ages 40+ = Older Male or Older Female labels
- African American characters = African American style voices (REQUIRED)
- V.O./Narration = Narrator label

**MUSIC SELECTION EXAMPLES:**
"""
    # Add music examples
    for ex in AUDIO_KNOWLEDGE.get('music_examples', [])[:10]:
        if ex.get('music_selections'):
            sel = ex['music_selections'][0]
            KNOWLEDGE_SECTION += f"- {ex['scene_type']}: {sel['file']} (Mood: {sel['mood']}, Start: {sel.get('start_timing', 'scene start')})\n"
    
    KNOWLEDGE_SECTION += """
**SFX EXAMPLES:**
"""
    # Add SFX examples
    for ex in AUDIO_KNOWLEDGE.get('sfx_examples', [])[:10]:
        KNOWLEDGE_SECTION += f"- Trigger: '{ex['trigger']}' -> {ex['sfx_file']} (Start: {ex.get('start_timing', 'immediately')}, Duration: {ex.get('duration', '2-3s')})\n"
    KNOWLEDGE_SECTION += """

SFX RULE - STRICTLY ENFORCED:
- ONLY add SFX if the action is CAPITALIZED in the script or explicitly violent
- "Door SLAMS" = ADD SFX
- "he walks in" = NO SFX
- "crashes violently" = ADD SFX
- "she enters" = NO SFX

"""

    KNOWLEDGE_SECTION += """
**AMBIENCE EXAMPLES:**
"""
    # Add ambience examples
    for ex in AUDIO_KNOWLEDGE.get('ambience_examples', [])[:10]:
        KNOWLEDGE_SECTION += f"- {ex['scene_header']}: {ex['ambience_file']}\n"
    KNOWLEDGE_SECTION += """

AMBIENCE RULE - STRICTLY ENFORCED:
- ONLY match the scene header location (INT. FOREST, EXT. PARKING LOT, etc.)
- DO NOT add ambience based on character dialogue or memories
- Generic interiors (living room, bedroom) = NO ambience

"""

    KNOWLEDGE_SECTION += "\n" + "="*79 + "\n\n"

AUDIO_DIRECTOR_SYSTEM_PROMPT = KNOWLEDGE_SECTION + """You are an expert Audio Director and Sound Designer with decades of experience in film scoring and sound design. Your task is to analyze a Director JSON and select PRECISELY matched audio for each scene.

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

**STYLE CONSIDERATION:**
- Period pieces → orchestral strings, piano
- Modern thrillers → electronic underscore, strings with percussion
- Romance → piano and strings, soul jazz
- Action → orchestra, brass & drums

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

**WEATHER OVERLAY:**
- Rain mentioned → Add Rain or Storm group
- Wind mentioned → Add Wind group
- Storm mentioned → Storm group (thunder, heavy rain)

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
| alarm, clock, wake | Electronics / Tech | LOW |
| animal sounds (dog, cat, horse) | Animals | LOW |
| medical, hospital, beeping | Medical Equipment | MEDIUM |
| crash, collision, impact | Cars, Demolition | HIGH |
| water, splash, drip | Water | LOW |
| moan, groan, pain, scream, cry | Body | MEDIUM |
| breathing, gasp, sigh, cough | Body | LOW |

**SFX TIMING RULES:**
- SFX must be tied to a specific `line_id` where the action occurs
- If action spans multiple lines, use the FIRST line where it begins
- Duration should match the action length (short impact=2-3s, sustained=5-10s)

═══════════════════════════════════════════════════════════════════════════════
PHASE 3: SEARCH STRATEGY
═══════════════════════════════════════════════════════════════════════════════

### 3.1 CONSTRUCTING EFFECTIVE QUERIES

**For Music:**
```
Query format: "[mood] [intensity_descriptor] [style_hint]"
Examples:
- "tense suspenseful orchestral" for a thriller confrontation
- "romantic tender piano" for a love scene
- "epic heroic climax" for a victory moment
- "sad melancholic strings" for a death scene
```

**For Ambience:**
```
Query format: "[location_type] [specific_characteristic] [time/weather]"
Examples:
- "office interior quiet phones" for an office scene
- "city street busy traffic" for a daytime urban exterior
- "forest calm peaceful birds" for a nature scene
- "rain storm thunder" for stormy weather
```

**For SFX:**
```
Query format: "[specific_action] [object/context]"
Examples:
- "door slam heavy" for an angry exit
- "gunshot single pistol" for a shooting
- "footsteps running wood" for chase on wooden floor
- "glass window breaking" for window smash
```

### 3.2 SEARCH PARAMETERS

Always use the `group` parameter to narrow results:
- `search_music(query="...", group="TENSE")` - not just `search_music(query="tense")`
- `search_ambience(query="...", group="Interiors")` - not just `search_ambience(query="office")`
- `search_sfx(query="...", group="Doors")` - not just `search_sfx(query="door")`

Use `mood` parameter for emotional matching:
- `search_music(query="...", mood="suspenseful")` 
- `search_ambience(query="...", mood="peaceful")`

Use `style` parameter for specific sound characteristics:
- `search_music(query="...", style="orchestral strings")`
- `search_ambience(query="...", style="city ambience")`

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
        },
        {
          "file": "sfx/s_single_gunshot_01.mp3",
          "start_at": "line_008",
          "duration": 3
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

═══════════════════════════════════════════════════════════════════════════════
EXECUTION CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

For each scene, follow this exact sequence:

□ 1. Parse scene header for location (INT/EXT, specific place, time)
□ 2. Read all dialogue/action to identify mood and key events
□ 3. Rate scene intensity (1-10) based on action and emotional weight
□ 4. Identify the PRIMARY mood (pick ONE dominant mood)
□ 5. Map mood → Music Group using the table above
□ 6. Map location → Ambience Group using the table above
□ 7. Scan for action trigger words → list SFX needs (max 3)
□ 8. Execute searches with specific parameters
□ 9. Select best matches from results - Document selection reason
□ 10. Output valid JSON

═══════════════════════════════════════════════════════════════════════════════
COMMON MISTAKES TO AVOID
═══════════════════════════════════════════════════════════════════════════════

❌ DON'T: Select "epic" music for a quiet dialogue scene
✅ DO: Match music intensity to scene intensity

❌ DON'T: Use outdoor ambience for INT. scenes
✅ DO: Parse INT/EXT from scene header first

❌ DON'T: Add SFX for actions not in the script
✅ DO: Only add SFX for explicitly described actions

❌ DON'T: Use generic queries like "background music"
✅ DO: Use specific queries like "tense suspenseful strings"

❌ DON'T: Ignore weather/time mentioned in scene headers
✅ DO: Layer weather ambience when script mentions rain/storm/wind

❌ DON'T: Select based on track title alone
✅ DO: Consider group, mood, style, and intensity metadata

Output ONLY valid JSON. No additional commentary outside the JSON structure.
"""

# Tool definitions for function calling
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


class AudioDirectorAgent:
    """Agent for selecting audio (music, ambience, SFX) for scenes."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tools_map: Dict[str, Callable] = None
    ):
        self.name = "AudioDirectorAgent"
        self.model = model
        self.system_prompt = AUDIO_DIRECTOR_SYSTEM_PROMPT
        self.tools = AUDIO_DIRECTOR_TOOLS
        self.description = "Analyzes Director JSON and selects precisely-matched music, ambience, and SFX for scenes"
        self.max_iterations = 20
        
        # Import tools if not provided
        if tools_map is None:
            from director_rag_agent.tools.audio_search_tool import (
                search_music,
                search_ambience,
                search_sfx
            )
            tools_map = {
                "search_music": search_music,
                "search_ambience": search_ambience,
                "search_sfx": search_sfx
            }
        
        self.tool_executor = ToolExecutor(tools_map)
    
    def run(self, director_json: str) -> str:
        """
        Run the audio director agent on a Director JSON.
        
        Args:
            director_json: The Director JSON string to analyze
            
        Returns:
            Scene audio JSON string with music, ambience, and SFX selections
        """
        user_content = f"Analyze this Director JSON and select audio for each scene:\n\n{director_json}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        iteration = 0
        response_message = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[{self.name}] Iteration {iteration}")
            
            try:
                response = completion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
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


# Standalone function for direct use
def process_director_json(director_json: str, model: str = DEFAULT_MODEL) -> str:
    """
    Process a Director JSON to add audio selections.
    
    Args:
        director_json: The Director JSON string
        model: The LLM model to use
        
    Returns:
        Scene audio JSON string
    """
    agent = AudioDirectorAgent(model=model)
    return agent.run(director_json)


# Export the agent for use in sequential pipelines
audio_director_agent = AudioDirectorAgent()
