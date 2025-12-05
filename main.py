
from datetime import datetime
import json
import re
import logging
import uuid
import uvicorn
from typing import Dict, List, Any, Optional
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# --- Import your custom modules ---
# Ensure these are in your python path
from director_rag_agent.agent import root_agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from director_rag_agent.tools.supabase_tool import save_to_supabase

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Director RAG Agent API",
    description="API to generate director scenes and audio metadata from scripts using GenAI.",
    version="1.0.0"
)

# --- Pydantic Models ---
class DirectorRequest(BaseModel):
    project_id: str = Field(..., description="The unique identifier for the project.")
    script_text: str = Field(..., description="The raw script text to be processed.")

class DirectorResponse(BaseModel):
    project_id: str
    success: bool
    message: Optional[str]
    scene_count: Optional[int]
    processing_time: Optional[float]
# --- Helper Functions ---

def clean_json_string(json_str: str) -> Dict[str, Any]:
    """Cleans Markdown code blocks from the LLM response to ensure valid JSON parsing."""
    cleaned = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    clean_text = cleaned.strip()
    return json.loads(clean_text)

def combine_output(director_json: Dict, scene_audio: Dict) -> Dict:
    """Merges scene visual data with audio data."""
    # Safety check for list lengths matching
    min_len = min(len(director_json.get('scenes', [])), len(scene_audio.get('scenes', [])))
    
    for index in range(min_len):
        director_json['scenes'][index]['music'] = scene_audio['scenes'][index].get('music')
        director_json['scenes'][index]['ambience'] = scene_audio['scenes'][index].get('ambience')
        director_json['scenes'][index]['sfx'] = scene_audio['scenes'][index].get('sfx')
    
    return director_json

def validate_voice_consistency(director_data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Checks character voice consistency."""
    character_registry: Dict[str, str] = {}
    errors: List[str] = []
    
    scenes = director_data.get("scenes", [])
    for scene in scenes:
        scene_id = scene.get("scene_id", "Unknown Scene")
        dialogue_lines = scene.get("dialogue", [])
        
        for line in dialogue_lines:
            char_name = line.get("character")
            voice_key = line.get("voice_key")
            
            if not char_name:
                continue
            
            if char_name in character_registry:
                registered_voice = character_registry[char_name]
                if voice_key != registered_voice:
                    errors.append(
                        f"Consistency Error in {scene_id}: Character '{char_name}' assigned '{voice_key}', expected '{registered_voice}'."
                    )
            else:
                character_registry[char_name] = voice_key

    return len(errors) == 0, errors

# --- Core Logic (Refactored for Async) ---

async def get_director_json(project_id: str, script_text: str, agent: Any) -> Dict[str, Any]:
    """
    Runs the Google ADK Runner asynchronously.
    Refactored to remove asyncio.run() so it plays nicely with FastAPI's event loop.
    """
    logger.info(f"Starting Director processing for project: {project_id}")

    input_payload = json.dumps({
        "project_id": project_id,
        "script_text": script_text
    })

    # 1. Initialize Runner
    runner = InMemoryRunner(agent=agent)
    
    # 2. Define IDs
    session_id = str(uuid.uuid4())
    user_id = "director_user"

    # 3. Create Session
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )

    # 4. Prepare the message
    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text=input_payload)]
    )

    # 5. Execute the Agent
    # We await the generator directly inside the FastAPI loop
    async for event in runner.run_async(
        session_id=session_id,
        user_id=user_id,
        new_message=user_msg
    ):
        pass 

    # 6. Retrieve the Session to get the State
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    # 7. Extract and Parse
    data = {}
    if session:
        try:
            director_str = session.state.get("director_json")
            audio_str = session.state.get("scene_audio_json")
            
            if not director_str or not audio_str:
                raise ValueError("Agent completed but output state keys are empty.")

            director_json = clean_json_string(director_str)
            scene_audio = clean_json_string(audio_str)
            data = combine_output(director_json, scene_audio)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {e}")
            raise ValueError(f"Failed to parse Agent output: {e}")
        except Exception as e:
            logger.error(f"State Extraction Error: {e}")
            raise ValueError(f"Error processing session state: {e}")
            
    if not data:
        raise ValueError("Director JSON generation failed.")

    return data

# --- API Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Director RAG Agent"}

@app.post("/generate-scenes", response_model=DirectorResponse)
async def generate_director_scenes_endpoint(request: DirectorRequest):
    """
    Generates director scenes from a script text.
    
    1. Runs the RAG Agent to analyze script.
    2. Validates voice consistency.
    3. Saves result to Supabase.
    """
    try:
        # 1. Generate Data
        start_time = time.time()
        director_data = await get_director_json(
            request.project_id, 
            request.script_text, 
            root_agent
        )
        
        # 2. Validate Data (Optional: Logic to handle validation failures)
        is_valid, warnings = validate_voice_consistency(director_data)
        
        if not is_valid:
            logger.warning(f"Voice consistency warnings for {request.project_id}: {warnings}")

        # 3. Save to Supabase
        # Assuming save_to_supabase returns the saved object or confirmation
        saved_record = await save_to_supabase(request.project_id, director_data)
        end_time = 0
        processing_time = round(end_time - start_time, 2)

        return DirectorResponse(
            project_id=request.project_id,
            success=saved_record['success'],
            message=saved_record['message'],
            scene_count=len(director_data['scenes']) if director_data['scenes'] else 0,
            processing_time=processing_time
        )

    except ValueError as ve:
        logger.error(f"Validation/Processing Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"System Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during processing.")

# --- Entry Point for Debugging ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)