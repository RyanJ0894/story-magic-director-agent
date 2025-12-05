# Copyright 2025
# File Upload Handler for Director Agent

"""
File Upload Handler - Upload and index knowledge base files at runtime.

This module provides tools for:
- Listing uploaded knowledge base files
- Indexing new files into the appropriate file search store
- Managing the file store configuration
"""

import os
import json
import time
import tempfile
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any
from google import genai
from google.adk.tools import FunctionTool, ToolContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration ---
CONFIG_PATH = Path(__file__).parent.parent / "file_store_config.json"


def load_config() -> Dict[str, Any]:
    """Load the file store configuration."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]):
    """Save configuration to file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def get_genai_client():
    """Get the Google GenAI client."""
    api_key = os.getenv('FILE_SEARCH_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Neither FILE_SEARCH_API_KEY nor GOOGLE_API_KEY found in environment")
    return genai.Client(api_key=api_key)


def generate_filename_from_content(file_data: bytes, mime_type: str) -> str:
    """Generate a unique filename based on file content hash."""
    content_hash = hashlib.md5(file_data).hexdigest()[:12]
    
    ext_map = {
        'application/json': 'json',
        'text/plain': 'txt',
        'text/csv': 'csv',
    }
    
    if '/' in mime_type:
        mime_ext = mime_type.split('/')[-1]
        if mime_ext in ['json', 'txt', 'csv']:
            ext = mime_ext
        else:
            ext = ext_map.get(mime_type, mime_ext)
    else:
        ext = ext_map.get(mime_type, 'json')
    
    return f"kb_{content_hash}.{ext}"


def create_file_search_store(client, display_name: str) -> str:
    """Create a new file search store."""
    store = client.file_search_stores.create(
        config={'display_name': display_name}
    )
    logger.info(f"Created new file search store: {store.name}")
    return store.name


async def index_knowledge_base_file(
    filename: str,
    store_type: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Index a knowledge base file into the appropriate file search store.
    
    Use this tool to add new knowledge base files (voices, music, ambience, SFX)
    to the search system.
    
    Args:
        filename: The name of the uploaded file
        store_type: Type of knowledge base - must be one of: "voices", "music", "ambience", "sfx"
        tool_context: Tool context (automatically provided by ADK)
    
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - message: Description of what happened
        - filename: The indexed filename
        - store_name: The file search store name
    """
    logger.info(f"[FileUpload] Indexing {filename} as {store_type}")
    
    # Validate store type
    valid_types = ["voices", "music", "ambience", "sfx"]
    if store_type not in valid_types:
        return {
            "status": "error",
            "message": f"Invalid store_type '{store_type}'. Must be one of: {valid_types}",
            "filename": filename
        }
    
    try:
        # Get API key
        api_key = os.getenv('FILE_SEARCH_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return {
                "status": "error",
                "message": "No API key found in environment",
                "filename": filename
            }
        
        # Load config
        config = load_config()
        
        # Get or create store for this type
        store_key = f"{store_type}_store_name"
        store_name = config.get(store_key)
        
        if not store_name:
            # Create new store
            client = get_genai_client()
            display_name = f"director-agent-{store_type}-store"
            store_name = create_file_search_store(client, display_name)
            config[store_key] = store_name
            save_config(config)
        
        # Try to load the artifact
        artifact_part = await tool_context.load_artifact(filename)
        
        file_data = None
        mime_type = None
        
        if artifact_part and artifact_part.inline_data:
            file_data = artifact_part.inline_data.data
            mime_type = artifact_part.inline_data.mime_type
        else:
            # Check session history for inline uploads
            if hasattr(tool_context, '_invocation_context'):
                ctx = tool_context._invocation_context
                if ctx and ctx.session and ctx.session.events:
                    for event in reversed(ctx.session.events[-5:]):
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if part.inline_data:
                                    file_data = part.inline_data.data
                                    mime_type = part.inline_data.mime_type
                                    break
                        if file_data:
                            break
        
        if not file_data or not mime_type:
            available = await tool_context.list_artifacts()
            return {
                "status": "error",
                "message": f"File '{filename}' not found. Available: {available}",
                "filename": filename
            }
        
        # Generate filename and save temporarily
        actual_filename = generate_filename_from_content(file_data, mime_type)
        
        # Check if already indexed
        indexed_files = config.get(f'{store_type}_indexed_files', [])
        if actual_filename in indexed_files:
            return {
                "status": "already_indexed",
                "message": f"File already indexed as {actual_filename}",
                "filename": actual_filename,
                "store_name": store_name
            }
        
        # Save to temp file
        temp_dir = Path(tempfile.gettempdir()) / "director_uploads"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / actual_filename
        
        with open(temp_file, 'wb') as f:
            f.write(file_data)
        
        # Upload to file search store
        client = get_genai_client()
        
        logger.info(f"[FileUpload] Uploading to store: {store_name}")
        upload_op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=str(temp_file)
        )
        
        # Wait for upload
        timeout = 120
        elapsed = 0
        while not upload_op.done and elapsed < timeout:
            time.sleep(3)
            elapsed += 3
            upload_op = client.operations.get(upload_op)
        
        if not upload_op.done:
            return {
                "status": "error",
                "message": f"Upload timed out after {timeout}s",
                "filename": actual_filename
            }
        
        # Update config
        indexed_files.append(actual_filename)
        config[f'{store_type}_indexed_files'] = indexed_files
        save_config(config)
        
        return {
            "status": "success",
            "message": f"Successfully indexed '{actual_filename}' into {store_type} store",
            "filename": actual_filename,
            "store_name": store_name,
            "store_type": store_type
        }
        
    except Exception as e:
        error_msg = f"Failed to index file: {str(e)}"
        logger.error(f"[FileUpload] {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "filename": filename
        }


async def list_knowledge_base_files(
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    List all knowledge base files and their indexing status.
    
    Args:
        tool_context: Tool context (automatically provided by ADK)
    
    Returns:
        A dictionary containing:
        - status: "success"
        - stores: Dict of store types with their indexed files
        - available_uploads: List of files available for indexing
    """
    logger.info("[FileUpload] Listing knowledge base files")
    
    try:
        # Get available artifacts
        available = await tool_context.list_artifacts()
        
        # Load config
        config = load_config()
        
        # Build stores info
        stores = {}
        for store_type in ["voices", "music", "ambience", "sfx"]:
            store_key = f"{store_type}_store_name"
            indexed_key = f"{store_type}_indexed_files"
            
            stores[store_type] = {
                "store_name": config.get(store_key, "NOT CONFIGURED"),
                "indexed_files": config.get(indexed_key, []),
                "configured": store_key in config
            }
        
        return {
            "status": "success",
            "stores": stores,
            "available_uploads": available,
            "message": f"Found {len(available)} available files"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list files: {str(e)}",
            "stores": {},
            "available_uploads": []
        }


# Create Function Tools
index_knowledge_base_tool = FunctionTool(index_knowledge_base_file)
list_knowledge_bases_tool = FunctionTool(list_knowledge_base_files)