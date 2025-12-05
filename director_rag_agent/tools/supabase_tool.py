# Copyright 2025
# Supabase Tool for Director Agent - Save Director JSON

"""
Supabase Tool - Save generated Director JSON to Supabase

This tool makes a POST request to the Supabase edge function
to store the generated Director JSON for a project.

NOTE: Google ADK does not support Union types in function parameters.
The director_json parameter must be a dict. If a string is passed,
it will be parsed internally.
"""

import os
import json
import logging
import httpx
from typing import Dict, Any
from google.adk.tools import FunctionTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Supabase Configuration ---
# These can be overridden via environment variables

SUPABASE_CALLBACK_URL = os.getenv(
    "SUPABASE_CALLBACK_URL",
    ""
)

SUPABASE_API_KEY = os.getenv(
    "SUPABASE_API_KEY",
    ""
)


async def save_to_supabase(
    project_id: str,
    director_json: dict,
) -> Dict[str, Any]:
    """
    Save the generated Director JSON to Supabase.
    
    After generating the complete Director JSON, ALWAYS call this tool
    to persist the data. This is REQUIRED for every execution.
    
    Args:
        project_id: The unique project identifier (UUID)
        director_json: The complete Director JSON object containing all scenes,
                      dialogue, music, ambience, and SFX. Must be a dictionary.
    
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - message: Description of what happened
        - project_id: The project ID that was saved
    """
    logger.info(f"[Supabase] Saving Director JSON for project: {project_id}")
    
    # Validate inputs
    if not project_id:
        return {
            "status": "error",
            "message": "project_id is required",
            "project_id": "",
            "success": False
        }
    
    if not director_json:
        return {
            "status": "error",
            "message": "director_json is required",
            "project_id": project_id,
            "success": False

        }
    
    # Validate director_json structure
    if not isinstance(director_json, dict):
        return {
            "status": "error",
            "message": "director_json must be a JSON object (dict)",
            "project_id": project_id,
            "success": False
        }
    
    # Check for required fields
    if 'scenes' not in director_json:
        return {
            "status": "error",
            "message": "director_json must contain 'scenes' array",
            "project_id": project_id,
            "success": False
        }
    
    # Prepare request payload
    payload = {
        "project_id": project_id,
        "director_json": director_json,
    }
    
    # Prepare headers
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Retry logic
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[Supabase] Attempt {attempt + 1}/{max_retries} - Sending to {SUPABASE_CALLBACK_URL}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    SUPABASE_CALLBACK_URL,
                    json=payload,
                    headers=headers
                )
                
                logger.info(f"[Supabase] Response status: {response.status_code}")
                
                if response.status_code in [200, 201]:
                    # Success
                    response_data = response.json() if response.text else {}
                    
                    return {
                        "status": "success",
                        "message": f"Successfully saved Director JSON for project {project_id}",
                        "project_id": project_id,
                        "scenes_saved": len(director_json.get('scenes', [])),
                        "response": response_data,
                        "success": True
                    }
                
                elif response.status_code == 401:
                    return {
                        "status": "error",
                        "message": "Authentication failed - check API key",
                        "project_id": project_id,
                        "success": False
                    }
                
                elif response.status_code == 400:
                    error_detail = response.text
                    return {
                        "status": "error",
                        "message": f"Bad request: {error_detail}",
                        "project_id": project_id,
                        "success": False
                    }
                
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"[Supabase] {last_error}")
                    
        except httpx.TimeoutException:
            last_error = "Request timed out"
            logger.warning(f"[Supabase] Timeout on attempt {attempt + 1}")
            
        except httpx.RequestError as e:
            last_error = f"Request error: {str(e)}"
            logger.warning(f"[Supabase] Request error on attempt {attempt + 1}: {e}")
            
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            logger.error(f"[Supabase] Unexpected error: {e}")
    
    # All retries failed
    return {
        "status": "error",
        "message": f"Failed after {max_retries} attempts. Last error: {last_error}",
        "project_id": project_id,
        "success": False
    }


def save_to_supabase_sync(
    project_id: str,
    director_json: dict,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for save_to_supabase.
    
    This is a synchronous version that can be used when async is not available.
    """
    import asyncio
    
    # Try to get existing event loop, or create a new one
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, we need to use different approach
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(
            save_to_supabase(project_id, director_json)
        )
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(
            save_to_supabase(project_id, director_json)
        )


# Create the FunctionTool
# Use the async version as ADK supports async tools
save_to_supabase_tool = FunctionTool(save_to_supabase)