
# from datetime import datetime
# import json
# import re
# import logging
# import uuid
# import uvicorn
# from typing import Dict, List, Any, Optional
# import time
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from pydantic import BaseModel, Field
# from fastapi.middleware.cors import CORSMiddleware

# # --- Import your custom modules ---
# # Ensure these are in your python path
# from director_rag_agent.agent import root_agent
# from google.adk.runners import InMemoryRunner
# from google.genai import types
# from director_rag_agent.tools.supabase_tool import save_to_supabase

# # --- Configuration ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="Director RAG Agent API",
#     description="API to generate director scenes and audio metadata from scripts using GenAI.",
#     version="1.0.0"
# )

# app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

# # --- Pydantic Models ---
# class DirectorRequest(BaseModel):
#     project_id: str = Field(..., description="The unique identifier for the project.")
#     script_text: str = Field(..., description="The raw script text to be processed.")

# class DirectorResponse(BaseModel):
#     project_id: str
#     success: bool
#     message: Optional[str]
#     scene_count: Optional[int]
#     processing_time: Optional[float]
# # --- Helper Functions ---

# def clean_json_string(json_str: str) -> Dict[str, Any]:
#     """Cleans Markdown code blocks from the LLM response to ensure valid JSON parsing."""
#     cleaned = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
#     cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
#     cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
#     clean_text = cleaned.strip()
#     return json.loads(clean_text)

# def combine_output(director_json: Dict, scene_audio: Dict) -> Dict:
#     """Merges scene visual data with audio data."""
#     # Safety check for list lengths matching
#     min_len = min(len(director_json.get('scenes', [])), len(scene_audio.get('scenes', [])))
    
#     for index in range(min_len):
#         director_json['scenes'][index]['music'] = scene_audio['scenes'][index].get('music')
#         director_json['scenes'][index]['ambience'] = scene_audio['scenes'][index].get('ambience')
#         director_json['scenes'][index]['sfx'] = scene_audio['scenes'][index].get('sfx')
    
#     return director_json

# def validate_voice_consistency(director_data: Dict[str, Any]) -> tuple[bool, List[str]]:
#     """Checks character voice consistency."""
#     character_registry: Dict[str, str] = {}
#     errors: List[str] = []
    
#     scenes = director_data.get("scenes", [])
#     for scene in scenes:
#         scene_id = scene.get("scene_id", "Unknown Scene")
#         dialogue_lines = scene.get("dialogue", [])
        
#         for line in dialogue_lines:
#             char_name = line.get("character")
#             voice_key = line.get("voice_key")
            
#             if not char_name:
#                 continue
            
#             if char_name in character_registry:
#                 registered_voice = character_registry[char_name]
#                 if voice_key != registered_voice:
#                     errors.append(
#                         f"Consistency Error in {scene_id}: Character '{char_name}' assigned '{voice_key}', expected '{registered_voice}'."
#                     )
#             else:
#                 character_registry[char_name] = voice_key

#     return len(errors) == 0, errors

# # --- Core Logic (Refactored for Async) ---

# async def get_director_json(project_id: str, script_text: str, agent: Any) -> Dict[str, Any]:
#     """
#     Runs the Google ADK Runner asynchronously.
#     Refactored to remove asyncio.run() so it plays nicely with FastAPI's event loop.
#     """
#     logger.info(f"Starting Director processing for project: {project_id}")

#     input_payload = json.dumps({
#         "project_id": project_id,
#         "script_text": script_text
#     })

#     # 1. Initialize Runner
#     runner = InMemoryRunner(agent=agent)
    
#     # 2. Define IDs
#     session_id = str(uuid.uuid4())
#     user_id = "director_user"

#     # 3. Create Session
#     await runner.session_service.create_session(
#         app_name=runner.app_name,
#         user_id=user_id,
#         session_id=session_id
#     )

#     # 4. Prepare the message
#     user_msg = types.Content(
#         role="user",
#         parts=[types.Part.from_text(text=input_payload)]
#     )

#     # 5. Execute the Agent
#     # We await the generator directly inside the FastAPI loop
#     async for event in runner.run_async(
#         session_id=session_id,
#         user_id=user_id,
#         new_message=user_msg
#     ):
#         pass 

#     # 6. Retrieve the Session to get the State
#     session = await runner.session_service.get_session(
#         app_name=runner.app_name,
#         user_id=user_id,
#         session_id=session_id
#     )
    
#     # In your get_director_json function, after getting the session:
#     if session:
#         logger.info(f"Session state keys: {session.state.keys() if session.state else 'No state'}")
#         logger.info(f"Full session state: {session.state}")
        
#     # 7. Extract and Parse
#     data = {}
#     if session:
#         try:
            
#             logger.info(f"Full session state: {json.dumps(session.state, default=str)}")
        
#             # Try different access patterns
#             director_str = (
#                 session.state.get("director_json") or 
#                 session.state.get("DirectorAgent", {}).get("output") or
#                 session.state.get("output", {}).get("director_json")
#             )
#             audio_str = (
#                 session.state.get("scene_audio_json") or
#                 session.state.get("AudioDirectorAgent", {}).get("output") or
#                 session.state.get("output", {}).get("scene_audio_json")
#             )
            
#             # director_str = session.state.get("director_json")
#             # audio_str = session.state.get("scene_audio_json")
            
#             if not director_str or not audio_str:
#                 raise ValueError("Agent completed but output state keys are empty.")

#             director_json = clean_json_string(director_str)
#             scene_audio = clean_json_string(audio_str)
#             data = combine_output(director_json, scene_audio)
#         except json.JSONDecodeError as e:
#             logger.error(f"JSON Parse Error: {e}")
#             raise ValueError(f"Failed to parse Agent output: {e}")
#         except Exception as e:
#             logger.error(f"State Extraction Error: {e}")
#             raise ValueError(f"Error processing session state: {e}")
            
#     if not data:
#         raise ValueError("Director JSON generation failed.")

#     return data

# # --- API Endpoints ---

# @app.get("/")
# def health_check():
#     return {"status": "ok", "service": "Director RAG Agent"}

# @app.post("/generate-scenes", response_model=DirectorResponse)
# async def generate_director_scenes_endpoint(request: DirectorRequest):
#     """
#     Generates director scenes from a script text.
    
#     1. Runs the RAG Agent to analyze script.
#     2. Validates voice consistency.
#     3. Saves result to Supabase.
#     """
#     try:
#         # 1. Generate Data
#         start_time = time.time()
#         director_data = await get_director_json(
#             request.project_id, 
#             request.script_text, 
#             root_agent
#         )
        
#         # 2. Validate Data (Optional: Logic to handle validation failures)
#         is_valid, warnings = validate_voice_consistency(director_data)
        
#         if not is_valid:
#             logger.warning(f"Voice consistency warnings for {request.project_id}: {warnings}")

#         # 3. Save to Supabase
#         # Assuming save_to_supabase returns the saved object or confirmation
#         saved_record = await save_to_supabase(request.project_id, director_data)
#         end_time = time.time()
#         processing_time = round(end_time - start_time, 2)

#         return DirectorResponse(
#             project_id=request.project_id,
#             success=saved_record['success'],
#             message=saved_record['message'],
#             scene_count=len(director_data['scenes']) if director_data['scenes'] else 0,
#             processing_time=processing_time
#         )

#     except ValueError as ve:
#         logger.error(f"Validation/Processing Error: {ve}")
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         logger.error(f"System Error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal Server Error during processing.")

# # --- Entry Point for Debugging ---
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Director RAG Agent API

FastAPI server for generating director scenes and audio metadata from scripts.
Uses LiteLLM with Groq for LLM inference.
"""

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
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# --- Import your custom modules ---
from director_rag_agent_2.agent import (
    DirectorAgent,
    AudioDirectorAgent,
    SequentialAgent,
    create_director_pipeline
)
from director_rag_agent.tools.supabase_tool import save_to_supabase

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for running sync LLM calls
executor = ThreadPoolExecutor(max_workers=4)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

app = FastAPI(
    title="Director RAG Agent API",
    description="API to generate director scenes and audio metadata from scripts using LiteLLM with Groq.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: str


# --- Retry Decorator ---

def retry_sync(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY_SECONDS):
    """
    Decorator for synchronous functions that retries on failure.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Attempt {attempt}/{max_retries} for {func.__name__}")
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed for {func.__name__}: {str(e)}"
                    )
                    if attempt < max_retries:
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
            
            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


async def retry_async(
    coro_func,
    *args,
    max_retries: int = MAX_RETRIES,
    delay: float = RETRY_DELAY_SECONDS,
    **kwargs
):
    """
    Async retry wrapper for coroutine functions.
    """
    last_exception = None
    func_name = getattr(coro_func, '__name__', 'async_function')
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries} for {func_name}")
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed for {func_name}: {str(e)}"
            )
            if attempt < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
    
    logger.error(f"All {max_retries} attempts failed for {func_name}")
    raise last_exception


# --- Helper Functions ---

def clean_json_string(json_str: str) -> Dict[str, Any]:
    """Cleans Markdown code blocks from the LLM response to ensure valid JSON parsing."""
    if not json_str:
        raise ValueError("Empty JSON string provided")
    
    # Remove markdown code blocks
    cleaned = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    clean_text = cleaned.strip()
    
    # Try to find JSON object/array in the string
    # Look for the first { or [ and last } or ]
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', clean_text)
    if json_match:
        clean_text = json_match.group(1)
    
    return json.loads(clean_text)


def combine_output(director_json: Dict, scene_audio: Dict) -> Dict:
    """Merges scene visual data with audio data."""
    # Safety check for list lengths matching
    director_scenes = director_json.get('scenes', [])
    audio_scenes = scene_audio.get('scenes', [])
    min_len = min(len(director_scenes), len(audio_scenes))
    
    for index in range(min_len):
        director_json['scenes'][index]['music'] = audio_scenes[index].get('music')
        director_json['scenes'][index]['ambience'] = audio_scenes[index].get('ambience')
        director_json['scenes'][index]['sfx'] = audio_scenes[index].get('sfx')
    
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


# --- Core Logic ---

@retry_sync(max_retries=MAX_RETRIES, delay=RETRY_DELAY_SECONDS)
def run_director_pipeline_sync(project_id: str, script_text: str) -> Dict[str, Any]:
    """
    Runs the director pipeline synchronously with retry logic.
    This will be called in a thread pool from async context.
    """
    logger.info(f"Starting Director pipeline for project: {project_id}")
    
    input_payload = json.dumps({
        "project_id": project_id,
        "script_text": script_text
    })
    
    # Create the pipeline
    pipeline = create_director_pipeline()
    
    # Run the pipeline
    results = pipeline.run(input_payload)
    
    # Extract outputs
    director_str = results.get("DirectorAgent", "")
    audio_str = results.get("AudioDirectorAgent", "")
    
    logger.info(f"DirectorAgent output length: {len(director_str)}")
    logger.info(f"AudioDirectorAgent output length: {len(audio_str)}")
    
    if not director_str:
        raise ValueError("DirectorAgent produced no output")
    if not audio_str:
        raise ValueError("AudioDirectorAgent produced no output")
    
    # Parse JSON outputs
    director_json = clean_json_string(director_str)
    logger.info("clean director json info")
    
    scene_audio = clean_json_string(audio_str)
    logger.info("clean audio director info")
    
    
    # Combine outputs
    combined_data = combine_output(director_json, scene_audio)
    
    return combined_data


async def get_director_json(project_id: str, script_text: str) -> Dict[str, Any]:
    """
    Runs the LiteLLM-based pipeline asynchronously.
    Uses ThreadPoolExecutor to run sync LLM calls without blocking the event loop.
    The retry logic is handled by the @retry_sync decorator on run_director_pipeline_sync.
    """
    logger.info(f"Starting Director processing for project: {project_id}")
    
    loop = asyncio.get_event_loop()
    
    # Run the sync pipeline in a thread pool (retry is handled by decorator)
    data = await loop.run_in_executor(
        executor,
        run_director_pipeline_sync,
        project_id,
        script_text
    )
    
    if not data:
        raise ValueError("Director JSON generation failed.")
    
    return data


# --- Alternative: Fully Async Implementation ---

class AsyncDirectorPipeline:
    """
    Async wrapper for the director pipeline.
    Useful if you want more control over the async execution.
    """
    
    def __init__(
        self,
        director_model: str = "groq/llama-3.3-70b-versatile",
        audio_model: str = "groq/llama-3.3-70b-versatile",
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY_SECONDS
    ):
        self.director_model = director_model
        self.audio_model = audio_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def _run_agent_with_retry(
        self,
        agent,
        input_data: str,
        agent_name: str
    ) -> str:
        """Run an agent with retry logic."""
        loop = asyncio.get_event_loop()
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries} for {agent_name}")
                output = await loop.run_in_executor(
                    executor,
                    agent.run,
                    input_data
                )
                logger.info(f"{agent_name} completed, output length: {len(output)}")
                return output
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for {agent_name}: {str(e)}"
                )
                if attempt < self.max_retries:
                    logger.info(f"Retrying {agent_name} in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
        
        logger.error(f"All {self.max_retries} attempts failed for {agent_name}")
        raise last_exception
    
    async def process(self, project_id: str, script_text: str) -> Dict[str, Any]:
        """Process a script through the full pipeline asynchronously with retries."""
        input_payload = json.dumps({
            "project_id": project_id,
            "script_text": script_text
        })
        
        # Run DirectorAgent with retry
        director_agent = DirectorAgent(model=self.director_model)
        director_output = await self._run_agent_with_retry(
            director_agent,
            input_payload,
            "DirectorAgent"
        )
        
        # Run AudioDirectorAgent with retry
        audio_agent = AudioDirectorAgent(model=self.audio_model)
        audio_output = await self._run_agent_with_retry(
            audio_agent,
            director_output,
            "AudioDirectorAgent"
        )
        
        # Parse and combine
        director_json = clean_json_string(director_output)
        scene_audio = clean_json_string(audio_output)
        
        return combine_output(director_json, scene_audio)


# --- API Endpoints ---

@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="Director RAG Agent",
        version="2.0.0",
        model="groq/llama-3.3-70b-versatile"
    )


@app.get("/health")
def detailed_health_check():
    """Detailed health check with configuration info."""
    import os
    return {
        "status": "ok",
        "service": "Director RAG Agent",
        "version": "2.0.0",
        "config": {
            "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
            "director_model": os.getenv("DIRECTOR_MODEL", "groq/llama-3.3-70b-versatile"),
            "audio_model": os.getenv("AUDIO_DIRECTOR_MODEL", "groq/llama-3.3-70b-versatile"),
            "max_retries": MAX_RETRIES,
            "retry_delay_seconds": RETRY_DELAY_SECONDS,
        }
    }


@app.post("/generate-scenes", response_model=DirectorResponse)
async def generate_director_scenes_endpoint(request: DirectorRequest):
    """
    Generates director scenes from a script text.
    
    1. Runs the RAG Agent to analyze script (with 3 retries).
    2. Validates voice consistency.
    3. Saves result to Supabase.
    """
    try:
        # 1. Generate Data (retry logic is built into run_director_pipeline_sync)
        start_time = time.time()
        director_data = await get_director_json(
            request.project_id,
            request.script_text
        )
        
        # 2. Validate Data
        is_valid, warnings = validate_voice_consistency(director_data)
        
        if not is_valid:
            logger.warning(f"Voice consistency warnings for {request.project_id}: {warnings}")

        # save to supabase
        saved_record = await save_to_supabase(request.project_id, director_data)
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        return DirectorResponse(
            project_id=request.project_id,
            success=saved_record.get('success', True),
            message=saved_record.get('message', 'Successfully processed'),
            scene_count=len(director_data.get('scenes', [])),
            processing_time=processing_time
        )

    except ValueError as ve:
        logger.error(f"Validation/Processing Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except json.JSONDecodeError as je:
        logger.error(f"JSON Parse Error: {je}")
        raise HTTPException(status_code=400, detail=f"Failed to parse agent output: {je}")
    except Exception as e:
        logger.error(f"System Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during processing.")


@app.post("/generate-scenes-preview")
async def generate_scenes_preview(request: DirectorRequest):
    """
    Preview endpoint that returns the raw JSON without saving to Supabase.
    Useful for testing and debugging. Includes retry logic.
    """
    try:
        start_time = time.time()
        director_data = await get_director_json(
            request.project_id,
            request.script_text
        )
        
        is_valid, warnings = validate_voice_consistency(director_data)
        end_time = time.time()
        
        return {
            "project_id": request.project_id,
            "success": True,
            "data": director_data,
            "validation": {
                "is_valid": is_valid,
                "warnings": warnings
            },
            "processing_time": round(end_time - start_time, 2)
        }

    except ValueError as ve:
        logger.error(f"Validation/Processing Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"System Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-director-only")
async def generate_director_only(request: DirectorRequest):
    """
    Run only the DirectorAgent (script parsing + voice assignment).
    Does not run AudioDirectorAgent. Includes retry logic.
    """
    try:
        start_time = time.time()
        
        input_payload = json.dumps({
            "project_id": request.project_id,
            "script_text": request.script_text
        })
        
        loop = asyncio.get_event_loop()
        director_agent = DirectorAgent()
        
        # Wrap in retry logic
        async def run_director():
            return await loop.run_in_executor(
                executor,
                director_agent.run,
                input_payload
            )
        
        director_output = await retry_async(
            run_director,
            max_retries=MAX_RETRIES,
            delay=RETRY_DELAY_SECONDS
        )
        
        director_json = clean_json_string(director_output)
        end_time = time.time()
        
        return {
            "project_id": request.project_id,
            "success": True,
            "data": director_json,
            "processing_time": round(end_time - start_time, 2)
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-audio-only")
async def generate_audio_only(director_json: Dict[str, Any]):
    """
    Run only the AudioDirectorAgent on existing Director JSON.
    Useful for re-processing audio without re-parsing the script.
    Includes retry logic.
    """
    try:
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        audio_agent = AudioDirectorAgent()
        
        # Wrap in retry logic
        async def run_audio():
            return await loop.run_in_executor(
                executor,
                audio_agent.run,
                json.dumps(director_json)
            )
        
        audio_output = await retry_async(
            run_audio,
            max_retries=MAX_RETRIES,
            delay=RETRY_DELAY_SECONDS
        )
        
        scene_audio = clean_json_string(audio_output)
        combined = combine_output(director_json, scene_audio)
        end_time = time.time()
        
        return {
            "success": True,
            "data": combined,
            "processing_time": round(end_time - start_time, 2)
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Startup/Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Verify configuration on startup."""
    import os
    
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set! API calls will fail.")
    else:
        logger.info("GROQ_API_KEY is configured")
    
    logger.info("Director RAG Agent API started")
    logger.info(f"Director Model: {os.getenv('DIRECTOR_MODEL', 'groq/llama-3.3-70b-versatile')}")
    logger.info(f"Audio Model: {os.getenv('AUDIO_DIRECTOR_MODEL', 'groq/llama-3.3-70b-versatile')}")
    logger.info(f"Retry Config: max_retries={MAX_RETRIES}, delay={RETRY_DELAY_SECONDS}s")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    executor.shutdown(wait=True)
    logger.info("Director RAG Agent API shutdown complete")


# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )