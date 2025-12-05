#!/usr/bin/env python3
# Copyright 2025
# Setup File Stores for Director Agent

"""
Setup File Stores - Upload and index knowledge base files into Google File Search stores.

This script creates separate file search stores for:
- Voice profiles (ElevenLabs voices)
- Music catalog
- Ambience catalog
- SFX catalog

Run this script once before using the Director Agent to set up the knowledge bases.

Usage:
    python setup_file_stores.py

Requirements:
    - GOOGLE_API_KEY or FILE_SEARCH_API_KEY environment variable set
    - Knowledge base JSON files in the knowledge_bases/ directory
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Paths ---
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_bases"
CONFIG_PATH = BASE_DIR / "file_store_config.json"


# --- Knowledge Base Files ---
KNOWLEDGE_BASES = {
    "voices": {
        "file": "eleven_labs_voices_current.json",
        "display_name": "director-agent-voices-store",
        "config_key": "voices_store_name"
    },
    "music": {
        "file": "music_catalog_full.json",
        "display_name": "director-agent-music-store",
        "config_key": "music_store_name"
    },
    "ambience": {
        "file": "ambience_catalog_enriched.json",
        "display_name": "director-agent-ambience-store",
        "config_key": "ambience_store_name"
    },
    "sfx": {
        "file": "sfx_catalog_enriched.json",
        "display_name": "director-agent-sfx-store",
        "config_key": "sfx_store_name"
    }
}


def get_api_key() -> str:
    """Get the Google API key from environment."""
    api_key = os.getenv('FILE_SEARCH_API_KEY') or os.getenv('GOOGLE_API_KEY')
    api_key = "AIzaSyDUBeAUnUv1DS_p32kLVA66di8TyuL0GY0"
    if not api_key:
        logger.error("No API key found!")
        logger.error("Please set either FILE_SEARCH_API_KEY or GOOGLE_API_KEY environment variable.")
        sys.exit(1)
    return api_key


def load_existing_config() -> dict:
    """Load existing configuration if it exists."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """Save configuration to file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {CONFIG_PATH}")


def create_file_search_store(client: genai.Client, display_name: str) -> str:
    """
    Create a new file search store.
    
    Args:
        client: The GenAI client
        display_name: Display name for the store
        
    Returns:
        The store name (resource ID)
    """
    logger.info(f"Creating file search store: {display_name}")
    
    store = client.file_search_stores.create(
        config={'display_name': display_name}
    )
    
    logger.info(f"Created store: {store.name}")
    return store.name or ""


def upload_file_to_store(client: genai.Client, store_name: str, file_path: Path) -> bool:
    """
    Upload a file to a file search store.
    
    Args:
        client: The GenAI client
        store_name: The store name to upload to
        file_path: Path to the file to upload
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Uploading {file_path.name} to store {store_name}")
    
    try:
        # Start the upload operation
        upload_op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=str(file_path)
        )
        
        # Wait for upload to complete
        timeout = 120  # 2 minutes
        elapsed = 0
        
        while not upload_op.done and elapsed < timeout:
            time.sleep(3)
            elapsed += 3
            upload_op = client.operations.get(upload_op)
            logger.info(f"  Uploading... {elapsed}s")
        
        if not upload_op.done:
            logger.error(f"  Upload timed out after {timeout}s")
            return False
        
        logger.info(f"  Upload complete!")
        return True
        
    except Exception as e:
        logger.error(f"  Upload failed: {e}")
        return False


def list_existing_stores(client: genai.Client) -> dict:
    """
    List existing file search stores.
    
    Returns:
        Dict mapping display names to store names
    """
    stores = {}
    try:
        for store in client.file_search_stores.list():
            display_name = getattr(store, 'display_name', None)
            if display_name:
                stores[display_name] = store.name
                logger.info(f"Found existing store: {display_name} -> {store.name}")
    except Exception as e:
        logger.warning(f"Could not list existing stores: {e}")
    return stores


def setup_knowledge_base(
    client: genai.Client,
    kb_type: str,
    kb_config: dict,
    existing_stores: dict,
    config: dict,
    force_recreate: bool = False
) -> bool:
    """
    Set up a single knowledge base store.
    
    Args:
        client: The GenAI client
        kb_type: Type of knowledge base (voices, music, etc.)
        kb_config: Configuration for this knowledge base
        existing_stores: Dict of existing stores
        config: Current config dict to update
        force_recreate: If True, recreate even if exists
        
    Returns:
        True if successful
    """
    display_name = kb_config["display_name"]
    config_key = kb_config["config_key"]
    file_name = kb_config["file"]
    file_path = KNOWLEDGE_BASE_DIR / file_name
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Setting up {kb_type} knowledge base")
    logger.info(f"{'='*60}")
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"Knowledge base file not found: {file_path}")
        logger.error(f"Please ensure {file_name} exists in the knowledge_bases/ directory")
        return False
    
    # Check if store already exists
    if display_name in existing_stores and not force_recreate:
        store_name = existing_stores[display_name]
        logger.info(f"Store already exists: {store_name}")
        config[config_key] = store_name
        return True
    
    # Create new store
    try:
        store_name = create_file_search_store(client, display_name)
        config[config_key] = store_name
        
        # Upload file to store
        success = upload_file_to_store(client, store_name, file_path)
        if not success:
            logger.error(f"Failed to upload {file_name}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up {kb_type}: {e}")
        return False


def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("Director Agent - Knowledge Base Setup")
    print("="*60 + "\n")
    
    # Get API key
    api_key = get_api_key()
    logger.info("API key found")
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    logger.info("GenAI client initialized")
    
    # Load existing config
    config = load_existing_config()
    
    # List existing stores
    logger.info("\nChecking for existing stores...")
    existing_stores = list_existing_stores(client)
    
    # Check for force recreate flag
    force_recreate = "--force" in sys.argv
    if force_recreate:
        logger.info("\n--force flag detected, will recreate all stores")
    
    # Setup each knowledge base
    all_success = True
    for kb_type, kb_config in KNOWLEDGE_BASES.items():
        success = setup_knowledge_base(
            client, kb_type, kb_config, existing_stores, config, force_recreate
        )
        if not success:
            all_success = False
    
    # Save configuration
    if config:
        config['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_config(config)
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    for kb_type, kb_config in KNOWLEDGE_BASES.items():
        config_key = kb_config["config_key"]
        store_name = config.get(config_key, "NOT CONFIGURED")
        status = "✓" if config_key in config else "✗"
        print(f"{status} {kb_type}: {store_name}")
    
    if all_success:
        print("\n✓ All knowledge bases configured successfully!")
        print(f"Configuration saved to: {CONFIG_PATH}")
    else:
        print("\n⚠ Some knowledge bases failed to configure.")
        print("Please check the errors above and try again.")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())