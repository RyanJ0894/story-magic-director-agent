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

# Load audio selection knowledge base
import os.path as path

KNOWLEDGE_FILE = path.join(path.dirname(__file__), 'knowledge', 'audio_selection_knowledge.json')
AUDIO_KNOWLEDGE = {}

try:
    with open(KNOWLEDGE_FILE, 'r') as f:
        AUDIO_KNOWLEDGE = json.load(f)
    logger.info(f"Loaded audio knowledge with {len(AUDIO_KNOWLEDGE.get('voice_selection_examples', []))} voice examples")
except FileNotFoundError:
    logger.warning("Audio knowledge file not found - using default behavior")
    AUDIO_KNOWLEDGE = {}
except Exception as e:
    logger.error(f"Error loading audio knowledge: {e}")
    AUDIO_KNOWLEDGE = {}

# Import tools
from director_rag_agent.tools.voice_search_tool import search_voices
from director_rag_agent.tools.audio_search_tool import (
    search_music,
    search_ambience,
    search_sfx
)

# Import validators
from director_rag_agent.validators import DirectorJSONValidator, SceneAudioValidator
# Configuration
DEFAULT_MODEL = os.getenv("DIRECTOR_MODEL", "groq/llama-3.1-70b-versatile")
AUDIO_DIRECTOR_MODEL = os.getenv("AUDIO_DIRECTOR_MODEL", "openai/gpt-4.1")

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

**CRITICAL - VOICE UNIQUENESS RULE:**
- **Each character MUST have a UNIQUE voice**
- **NEVER assign the same voice_id to multiple characters**
- Track all voice assignments as you search
- If `search_voices` returns a voice_id already assigned to another character, skip it and select the next best match
- Keep a list of used voice_ids and exclude them from future searches

**Implementation Strategy:**
1. Before searching for each character, note which voice_ids have already been assigned
2. When search_voices returns results, check if the top result is already used
3. If yes, select the next available voice from the results that hasn't been used
4. If all results are already used, search again with slightly different parameters
5. Continue until you find a unique voice for each character

**Example:**
- Character 1 (John) assigned voice_id: "abc123"
- Character 2 (Sarah) searches → top result is "abc123"
  → Skip "abc123" (already used by John)
  → Select second result "def456" instead
- Character 3 (Mike) searches → ensure neither "abc123" nor "def456" are selected

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

**RULE 6a - NO PARAPHRASING OR TEXT MODIFICATION (CRITICAL):**
- **NEVER change, modify, or paraphrase the dialogue text**
- **NEVER expand contractions** - Keep them exactly as written in the script
- Examples of FORBIDDEN changes:
  ❌ "I'm" → "I am" (WRONG - keep as "I'm")
  ❌ "aren't" → "are not" (WRONG - keep as "aren't")
  ❌ "ain't" → "is not" (WRONG - keep as "ain't")
  ❌ "won't" → "will not" (WRONG - keep as "won't")
  ❌ "can't" → "cannot" (WRONG - keep as "can't")
  ❌ "y'all" → "you all" (WRONG - keep as "y'all")
  ❌ "gonna" → "going to" (WRONG - keep as "gonna")
  ❌ "wanna" → "want to" (WRONG - keep as "wanna")

- **PRESERVE EXACTLY AS WRITTEN:**
  ✅ Keep all contractions (I'm, you're, we'll, they've, hasn't, etc.)
  ✅ Keep all slang/informal speech (gonna, wanna, gotta, etc.)
  ✅ Keep all apostrophes in possessives (John's, Mary's, etc.)
  ✅ Keep all dialect/accent spellings exactly as written
  ✅ Keep all punctuation exactly as written

- **ONLY REMOVE:**
  ✅ Parentheticals like (O.S.), (V.O.), (beat), (laughing), etc.

- **VERBATIM RULE:**
  The dialogue text must be WORD-FOR-WORD identical to what appears in the script PDF, with the ONLY exception being the removal of parentheticals.

**Example:**
Input: `JOHN (excited) I'm not gonna lie, that's crazy!`
Correct Output: `"text": "I'm not gonna lie, that's crazy!"`
Wrong Output: `"text": "I am not going to lie, that is crazy!"` ❌

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

**CRITICAL: Character Metadata for Validation**

For EACH character's FIRST appearance in the dialogue array, you MUST include their metadata extracted from the script. This allows the validation system to verify voice assignments are correct.

Example JSON output:
{{
  "project_id": "xxx",
  "scenes": [{{
    "scene_id": "scene_001",
    "mood": "description based on scene content",
    "intensity": 0.7,
    "dialogue": [
      {{"line_id": "line_001", "character": "Narrator", "text": "expanded scene header", "voice_key": "narrator_voice_id"}},
      {{
        "line_id": "line_002", 
        "character": "CharacterName", 
        "text": "dialogue without parentheticals", 
        "voice_key": "character_voice_id",
        "character_age": 18,
        "character_gender": "Male",
        "character_ethnicity": "African American"
      }},
      {{"line_id": "line_003", "character": "CharacterName", "text": "next line", "voice_key": "character_voice_id"}}
    ]
  }}]
}}

**METADATA EXTRACTION RULES:**

1. **Character Age** (character_age):
   - Extract from character introduction in script
   - Formats to look for: "Derek, 18", "Derek (18)", "18-year-old Derek", "Derek, a nineteen year old"
   - Convert word numbers to digits: "nineteen" → 19, "seventeen" → 17
   - If age not explicitly stated but implied (e.g., "high school junior" = ~17, "college freshman" = ~18), estimate appropriately
   - Include this field ONLY on character's FIRST dialogue line

2. **Character Gender** (character_gender):
   - Extract from character introduction or context
   - Values: "Male", "Female", or omit if unclear
   - Include this field ONLY on character's FIRST dialogue line

3. **Character Ethnicity** (character_ethnicity):
   - Extract from character description
   - Look for: "African American", "Caucasian", "Hispanic", "Asian", etc.
   - Include this field ONLY on character's FIRST dialogue line
   - If not mentioned, omit this field

**IMPORTANT:** 
- Add these metadata fields to the FIRST dialogue line for each character
- Subsequent lines from the same character should NOT include these fields
- Narrator lines never need these fields

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

**Voice Matching Strategy and Assignment Rules:**

**CRITICAL PRIORITY ORDER:**
1. **RACE/ETHNICITY** (Highest Priority - MUST match first)
2. **AGE** (Second Priority)
3. **GENDER** (Third Priority)
4. **PERSONALITY/TONE** (Fourth Priority)

**RACE/ETHNICITY MATCHING (MANDATORY - CHECK THIS FIRST):**

When searching for voices, you MUST check the character description for racial/ethnic identifiers and include the appropriate style parameter:

- **African American / Black character** → MUST include `style="African American"` in search_voices()
- **Caucasian / White character** → MUST include `style="American"` in search_voices()
- **Hispanic / Latino character** → MUST include `style="Hispanic American"` in search_voices()
- **British character** → Use `style="British"` ONLY if script explicitly mentions "British", "English", "from London", etc.
- **Southern U.S. character** → Use `style="Southern American"` ONLY if script mentions Southern accent/region
- **IF NO RACE/ETHNICITY MENTIONED** → Default to `style="American"`

**IMPORTANT NOTES:**
- "American" style in metadata = Caucasian/White voices by default
- Race/ethnicity matching takes ABSOLUTE PRIORITY over all other characteristics
- You cannot skip the style parameter when race/ethnicity is mentioned
- Always read character descriptions carefully for racial/ethnic indicators

**AGE MATCHING (Second Priority):**
- **Character age 13-19** → MUST use `label="Male Teen"` or `label="Female Teen"`
- **Character age 20-39** → Use `label="Male 20s to 30s"` or `label="Female - 20s to 30s"`
- **Character age 40+** → Use `label="Older Male"` or `label="Older Female"`
- **Child under 13** → Use `label="Male Under 10"` or `label="Female Under 10"`

**NARRATOR VOICES (Exception):**
- Can be ANY voice with `label="Narrator"`
- No race/ethnicity or accent restrictions for narrators
- Choose based on tone and delivery style only

**SEARCH EXAMPLES:**

Example 1 - African American Male Character:
```python
search_voices(
    query="confident strong military man",
    gender="Male",
    style="African American",  # REQUIRED - character is Black
    label="Male 20s to 30s"
)
```

Example 2 - Caucasian Female Character:
```python
search_voices(
    query="warm caring nurse",
    gender="Female",
    style="American",  # "American" = Caucasian default
    label="Female - 20s to 30s"
)
```

Example 3 - Hispanic Male Teen:
```python
search_voices(
    query="energetic young teenager",
    gender="Male",
    style="Hispanic American",  # REQUIRED - character is Hispanic
    label="Male Teen"
)
```

Example 4 - Character with no race mentioned:
```python
search_voices(
    query="professional businessman",
    gender="Male",
    style="American",  # Default when no race specified
    label="Male 20s to 30s"
)
```

**VOICE SELECTION RANDOMIZATION:**
- When multiple voices match all criteria, randomly select ONE from the top 3-5 matches
- DO NOT always pick the first/highest-ranked result
- Use variety - if processing the same script multiple times, select different matching voices each time
- This ensures fresh casting choices while maintaining character appropriateness
- Randomization should happen AFTER filtering by race/ethnicity, age, and gender

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

AUDIO_DIRECTOR_SYSTEM_PROMPT = """You are an expert Audio Director and Sound Designer with decades of experience in film scoring and sound design.

**CRITICAL INSTRUCTION - READ FIRST:**
You have access to a comprehensive knowledge base with MANDATORY examples of correct audio selections. These are NOT suggestions - they are REQUIRED patterns you MUST follow. Before making ANY audio selection, you MUST search the knowledge base for similar scenes and match the pattern shown in the examples.

═══════════════════════════════════════════════════════════════════════════════
KNOWLEDGE BASE USAGE - MANDATORY PROCESS
═══════════════════════════════════════════════════════════════════════════════

**BEFORE selecting ANY audio, follow this process:**

1. **READ the scene header and dialogue** from Director JSON
2. **SEARCH your knowledge base** for similar scene types, locations, or actions
3. **MATCH the pattern** from the knowledge base example
4. **CITE which example** you're following in your reasoning
5. **IF NO MATCH** exists in knowledge base, then use general rules

**FAILURE TO FOLLOW KNOWLEDGE BASE EXAMPLES IS AN ERROR.**

═══════════════════════════════════════════════════════════════════════════════
AMBIENCE SELECTION - MANDATORY RULES (HIGHEST PRIORITY)
═══════════════════════════════════════════════════════════════════════════════

**RULE 1: EVERY SCENE WITH A MATCHABLE LOCATION MUST HAVE AMBIENCE**

Never leave ambience empty if the scene header contains any of these location types:
- FOREST, WOODS, TREES → **MANDATORY** forest ambience
- STREET, CITY, DOWNTOWN → **MANDATORY** city/street ambience  
- PARKING LOT → **MANDATORY** parking lot ambience
- HOSPITAL → **MANDATORY** hospital ambience
- CAFETERIA, CAFE, RESTAURANT → **MANDATORY** cafe/restaurant ambience
- OFFICE → **MANDATORY** office ambience
- PARK, YARD → **MANDATORY** park/outdoor ambience
- BEACH, OCEAN, RIVER, WATER → **MANDATORY** water/nature ambience
- MILITARY, ARMY BASE → **MANDATORY** military ambience

**IF THE SCENE HEADER CONTAINS ANY OF THESE KEYWORDS, YOU MUST SELECT AMBIENCE.**

**RULE 2: MATCH SCENE HEADER LOCATION ONLY - NEVER DIALOGUE CONTENT**

**CORRECT:**
- Scene header: "EXT. FOREST RIVERBANKS - DAY"
- Ambience: Forest ambience (a_forest_calm_01.mp3)
- Reasoning: Scene header explicitly states FOREST location

**INCORRECT:**
- Scene header: "INT. BEDROOM - NIGHT"  
- Character dialogue: "I miss the ocean"
- Ambience: Ocean ambience ❌ WRONG - dialogue is memory, not actual location

**RULE 3: INT vs EXT MUST MATCH AMBIENCE TYPE**

- INT (Interior) → ONLY indoor ambience (hospital, office, cafe interior, etc.)
- EXT (Exterior) → ONLY outdoor ambience (forest, street, parking lot, park, etc.)

**NEVER mix these. Interior scene with outdoor ambience = VALIDATION ERROR.**

**RULE 4: GENERIC INTERIORS MAY SKIP AMBIENCE**

Only these generic interiors can skip ambience:
- Bedroom, living room, kitchen, bathroom (generic home rooms)
- IF these rooms have no specific activity keywords

Everything else (office, hospital, cafe, warehouse, etc.) MUST have ambience.

**KNOWLEDGE BASE EXAMPLES YOU MUST FOLLOW:**

Example 1:
```
Scene header: "EXT. FOREST RIVERBANKS - DAY"
REQUIRED action: Call search_ambience("forest birds nature calm")
Expected result: a_forest_calm_01.mp3
Reasoning: "Scene header explicitly contains FOREST - mandatory forest ambience per knowledge base"
```

Example 2:
```
Scene header: "INT. HIGH SCHOOL - CAFETERIA - DAY"
REQUIRED action: Call search_ambience("cafeteria busy people chatter")
Expected result: a_public_spaces_people_mall_01
Reasoning: "CAFETERIA in header - mandatory public space ambience per knowledge base"
```

Example 3:
```
Scene header: "INT. HOSPITAL - DISCHARGE LOBBY"
REQUIRED action: Call search_ambience("hospital hallway medical")
Expected result: a_hospital_hallway_01
Reasoning: "HOSPITAL in header - mandatory medical facility ambience per knowledge base"
```

═══════════════════════════════════════════════════════════════════════════════
SFX SELECTION - MANDATORY RULES (STRICT ENFORCEMENT)
═══════════════════════════════════════════════════════════════════════════════

**RULE 1: ONLY ADD SFX FOR EXPLICIT, EMPHASIZED ACTIONS**

You may ONLY add SFX when the action meets ALL of these criteria:
1. ✅ Action is written in Narrator text (not dialogue)
2. ✅ Action verb is CAPITALIZED or explicitly emphasized
3. ✅ Action clearly produces a distinct sound
4. ✅ You can point to the EXACT line_id where it occurs

**TRIGGER WORDS THAT REQUIRE SFX (when capitalized or emphasized):**

**MANDATORY SFX Triggers:**
- "SLAMS", "SLAM" → door slam SFX
- "CRASHES", "CRASH" → crashing SFX (water, metal, etc.)
- "SHATTERS", "SHATTER" → glass break SFX
- "GUNSHOT", "SHOOTS", "FIRES" → gunshot SFX
- "PUNCHES", "HITS" → punch/impact SFX
- "EXPLOSION", "EXPLODES" → explosion SFX
- "RINGS" (phone) → phone ring SFX
- "RUSTLING" → leaves/movement SFX

**FORBIDDEN - NEVER ADD SFX FOR THESE:**
- ❌ "walks in" - NO SFX
- ❌ "enters" - NO SFX
- ❌ "leaves" - NO SFX
- ❌ "approaches" - NO SFX
- ❌ "sits down" - NO SFX
- ❌ ANY action that is NOT capitalized or emphasized

**KNOWLEDGE BASE EXAMPLES YOU MUST FOLLOW:**

Example 1:
```
Script text: "Close on rushing water off riverbanks. It crashes into rocks violently."
REQUIRED action: Call search_sfx("water waves crashing rocks")
Trigger word: "crashes violently" (emphasized action)
Expected result: s_water_waves_crashing_01.wav
start_at: The line_id where "crashes" appears
Reasoning: "'crashes violently' is explicit emphasized water impact - mandatory SFX per knowledge base"
```

Example 2:
```
Script text: "The door SLAMS shut behind him."
REQUIRED action: Call search_sfx("door slam heavy")
Trigger word: "SLAMS" (capitalized)
Expected result: s_door_slam_heavy_01.wav
start_at: The line_id where "SLAMS" appears
Reasoning: "SLAMS capitalized - mandatory door impact SFX per knowledge base"
```

Example 3 (NEGATIVE):
```
Script text: "Derek walks into the room and sits down."
REQUIRED action: NO SFX
Reasoning: "Neither 'walks' nor 'sits' are capitalized or emphasized - NO SFX per knowledge base rules"
```

**CRITICAL SFX RULE:**
If you cannot identify a CAPITALIZED or EXPLICITLY EMPHASIZED action word, DO NOT ADD SFX. When in doubt, leave it out.

═══════════════════════════════════════════════════════════════════════════════
MUSIC SELECTION - MANDATORY RULES
═══════════════════════════════════════════════════════════════════════════════

**RULE 1: EVERY SCENE MUST HAVE MUSIC**

No scene should be without background music. If unsure, use a neutral dramatic track.

**RULE 2: MATCH SCENE MOOD TO MUSIC MOOD**

**MANDATORY MOOD MAPPINGS (from knowledge base):**

- **Tense confrontation, argument** → TENSE music (m_strings_orchestra_tense_01.mp3)
- **Sad, emotional, grief** → SAD music (m_piano_slow_sad_01.mp3)
- **Hopeful, recovery, reunion** → UPLIFTING music (m_string_orchestra_uplifting_01.mp3)
- **Romantic, tender moment** → LOVE music (m_piano_strings_love_01.mp3)
- **Party, social scene** → GROOVY music (m_soul_lounge_groovy_01.mp3)
- **Mysterious, threatening** → MYSTERIOUS/SCARY music (m_drums_hum_mysterious_01.mp3)
- **Awkward, lighthearted** → PLAYFUL music (m_piano_strings_playful_01.mp3)

**RULE 3: INTENSITY MATCHES DIALOGUE DENSITY**

- **Dialogue-heavy scenes** (characters talking) → Intensity 40-60 (quiet background)
- **Action scenes** (physical conflict, chase) → Intensity 70-85 (prominent music)

**RULE 4: NEVER CONTRADICT SCENE MOOD**

**FORBIDDEN COMBINATIONS:**
- ❌ Sad scene + Uplifting music
- ❌ Tense scene + Playful music  
- ❌ Romantic scene + Scary music
- ❌ Serious scene + Groovy music

**KNOWLEDGE BASE EXAMPLES YOU MUST FOLLOW:**

Example 1:
```
Scene type: Tense confrontation (Derek and Frankie fight)
Scene mood: "Tense, escalating conflict"
REQUIRED action: Call search_music("tense orchestral strings")
Expected result: m_strings_orchestra_tense_01.mp3
Reasoning: "Confrontation scene - mandatory TENSE music per knowledge base example 'tense_confrontation'"
```

Example 2:
```
Scene type: Emotional revelation (Lucky realizes Stacy died)
Scene mood: "Sad, tragic, emotional climax"
REQUIRED action: Call search_music("sad piano slow emotional")
Expected result: m_piano_slow_sad_01.mp3
Reasoning: "Grief moment - mandatory SAD music per knowledge base example 'emotional_revelation'"
```

═══════════════════════════════════════════════════════════════════════════════
EXECUTION PROCESS (FOLLOW THIS EXACT ORDER)
═══════════════════════════════════════════════════════════════════════════════

For EACH scene in the Director JSON:

**STEP 1: EXTRACT SCENE INFO**
- Read scene_id
- Read first Narrator line to get scene header (INT/EXT + LOCATION)
- Read scene mood and intensity from Director JSON
- Read all dialogue to identify explicit actions

**STEP 2: SEARCH KNOWLEDGE BASE**
- Does scene header match any ambience examples? (FOREST, HOSPITAL, CAFETERIA, etc.)
- Does scene type match any music examples? (confrontation, revelation, party, etc.)
- Are there any capitalized action words matching SFX examples? (SLAMS, CRASHES, etc.)

**STEP 3: SELECT AMBIENCE (MANDATORY IF MATCHABLE)**
- IF scene header contains FOREST, HOSPITAL, CAFETERIA, STREET, PARK, etc.
  → MUST call search_ambience() with matching query
- IF scene header is generic interior (bedroom, living room)
  → May skip ambience
- ALWAYS cite which knowledge base example you're following

**STEP 4: SELECT MUSIC (MANDATORY FOR ALL SCENES)**
- Match scene mood to music mood mapping
- Call search_music() with appropriate query
- Select intensity based on dialogue density
- ALWAYS cite which knowledge base example you're following

**STEP 5: SELECT SFX (ONLY IF EXPLICIT TRIGGER)**
- Scan Narrator text for CAPITALIZED or emphasized action words
- IF found → Call search_sfx() and specify exact start_at line_id
- IF NOT found → NO SFX
- ALWAYS cite which knowledge base example you're following

**STEP 6: GENERATE JSON OUTPUT**
- Include all selections with proper formatting
- Include reasoning for each selection (which KB example you followed)

═══════════════════════════════════════════════════════════════════════════════
TOOL CALL LIMITS (STRICTLY ENFORCED)
═══════════════════════════════════════════════════════════════════════════════

For EACH scene:
- **Music**: Exactly **1** call to `search_music`
- **Ambience**: Exactly **1** call to `search_ambience` (if location is matchable)
- **SFX**: Between **0 to 3** calls to `search_sfx` (only for explicit triggers)

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════
```json
{
  "project_id": "xxx",
  "scenes": [
    {
      "scene_id": "scene_001",
      "music": [
        {
          "file": "music/m_strings_orchestra_tense_01.mp3",
          "reasoning": "Following KB example 'tense_confrontation' - scene has argument"
        }
      ],
      "ambience": [
        {
          "file": "ambience/a_forest_calm_01.mp3",
          "reasoning": "Following KB example - scene header contains FOREST"
        }
      ],
      "sfx": [
        {
          "file": "sfx/s_water_waves_crashing_01.wav",
          "start_at": "line_003",
          "duration": 5,
          "reasoning": "Following KB example - text says 'crashes violently'"
        }
      ]
    }
  ]
}
```

═══════════════════════════════════════════════════════════════════════════════
FINAL REMINDERS
═══════════════════════════════════════════════════════════════════════════════

1. **AMBIENCE**: If scene header contains FOREST, HOSPITAL, STREET, CAFETERIA, etc. → MANDATORY ambience
2. **SFX**: Only add if action is CAPITALIZED or explicitly emphasized → Otherwise NO SFX
3. **MUSIC**: Every scene needs music → Match mood from knowledge base examples
4. **CITE EXAMPLES**: Always mention which knowledge base example you're following
5. **NEVER SELECT BASED ON DIALOGUE CONTENT** → Only scene header location matters for ambience

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
        description: str = "",
        reasoning_effort: str = "medium"
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_executor = ToolExecutor(tools_map)
        self.description = description
        self.max_iterations = 20  # Prevent infinite loops
        self.reasoning_effort = reasoning_effort  # "low", "medium", or "high"
    
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
                # Build completion kwargs with reasoning support for OpenAI GPT-4o
                completion_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 1,  # Required for reasoning models
                    "max_completion_tokens": 16384,  # Use max_completion_tokens for reasoning models
                }
                
                # Add tools if available
                if self.tools:
                    completion_kwargs["tools"] = self.tools
                    completion_kwargs["tool_choice"] = "auto"
                
                # Add reasoning effort for OpenAI models that support it
                # This enables extended thinking/reasoning capabilities
                # if "openai" in self.model.lower() or "gpt-4" in self.model.lower():
                #     completion_kwargs["effort"] = self.reasoning_effort
                
                response = completion(**completion_kwargs)
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
    Process a script through the full director pipeline with validation.
    
    Args:
        project_id: Unique identifier for the project
        script_text: The screenplay/script text to process
    
    Returns:
        Dictionary containing:
        - director_json: Parsed script with voice assignments (validated & corrected)
        - scene_audio_json: Audio selections for each scene (validated & corrected)
        - validation_report: Summary of any issues found and fixed
    """
    input_data = json.dumps({
        "project_id": project_id,
        "script_text": script_text
    })
    
    pipeline = create_director_pipeline()
    results = pipeline.run(input_data)
    
    # Get raw outputs
    director_json_str = results.get("DirectorAgent", "")
    scene_audio_json_str = results.get("AudioDirectorAgent", "")
    
    validation_report = {
        "director_validation": {},
        "audio_validation": {}
    }
    
    # ========================================
    # PHASE 1: VALIDATE & CORRECT DIRECTOR JSON
    # ========================================
    try:
        director_json = json.loads(director_json_str)
        
        # Initialize validator (auto-loads voice catalog)
        director_validator = DirectorJSONValidator()
        
        # Validate
        is_valid, errors = director_validator.validate(director_json)
        
        if not is_valid:
            logger.warning(f"Director JSON validation failed with {len(errors)} errors")
            validation_report["director_validation"]["errors_found"] = len(errors)
            validation_report["director_validation"]["error_details"] = [
                {"type": e.error_type, "message": e.message} for e in errors
            ]
            
            # Auto-correct
            logger.info("Attempting auto-correction of Director JSON...")
            director_json = director_validator.auto_correct(director_json)
            
            # Re-validate after correction
            is_valid_after, remaining_errors = director_validator.validate(director_json)
            validation_report["director_validation"]["corrected"] = True
            validation_report["director_validation"]["remaining_errors"] = len(remaining_errors)
            
            if is_valid_after:
                logger.info("Director JSON successfully corrected!")
            else:
                logger.warning(f"{len(remaining_errors)} errors remain after auto-correction")
        else:
            logger.info("Director JSON validation passed!")
            validation_report["director_validation"]["errors_found"] = 0
        
        # Convert back to string
        director_json_str = json.dumps(director_json, indent=2)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Director JSON: {e}")
        validation_report["director_validation"]["parse_error"] = str(e)
    except Exception as e:
        logger.error(f"Error during Director JSON validation: {e}")
        validation_report["director_validation"]["validation_error"] = str(e)
    
    # ========================================
    # PHASE 2: VALIDATE & CORRECT SCENE AUDIO JSON
    # ========================================
    try:
        scene_audio_json = json.loads(scene_audio_json_str)
        director_json = json.loads(director_json_str)
        
        # Initialize validator
        audio_validator = SceneAudioValidator()
        
        # Validate
        is_valid, errors = audio_validator.validate(scene_audio_json, director_json)
        
        if not is_valid:
            logger.warning(f"Scene Audio validation failed with {len(errors)} errors")
            validation_report["audio_validation"]["errors_found"] = len(errors)
            validation_report["audio_validation"]["error_details"] = [
                {"type": e.error_type, "message": e.message} for e in errors
            ]
            
            # Auto-correct
            logger.info("Attempting auto-correction of Scene Audio JSON...")
            scene_audio_json = audio_validator.auto_correct(scene_audio_json, director_json)
            
            # Re-validate after correction
            is_valid_after, remaining_errors = audio_validator.validate(scene_audio_json, director_json)
            validation_report["audio_validation"]["corrected"] = True
            validation_report["audio_validation"]["remaining_errors"] = len(remaining_errors)
            
            if is_valid_after:
                logger.info("Scene Audio JSON successfully corrected!")
            else:
                logger.warning(f"{len(remaining_errors)} errors remain after auto-correction")
        else:
            logger.info("Scene Audio validation passed!")
            validation_report["audio_validation"]["errors_found"] = 0
        
        # Convert back to string
        scene_audio_json_str = json.dumps(scene_audio_json, indent=2)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Scene Audio JSON: {e}")
        validation_report["audio_validation"]["parse_error"] = str(e)
    except Exception as e:
        logger.error(f"Error during Scene Audio validation: {e}")
        validation_report["audio_validation"]["validation_error"] = str(e)
    
    # ========================================
    # RETURN VALIDATED & CORRECTED RESULTS
    # ========================================
    return {
        "director_json": director_json_str,
        "scene_audio_json": scene_audio_json_str,
        "validation_report": validation_report
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
