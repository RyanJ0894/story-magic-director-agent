"""
Validation and Auto-Correction Layer

Validates agent outputs and automatically corrects common mistakes
to ensure consistency and reliability before audio generation.
"""

import logging
import json
import re
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a validation error with fix information."""
    
    def __init__(self, error_type: str, message: str, location: str, fix_data: Dict = None):
        self.error_type = error_type
        self.message = message
        self.location = location
        self.fix_data = fix_data or {}
    
    def __repr__(self):
        return f"ValidationError({self.error_type}: {self.message} at {self.location})"


class DirectorJSONValidator:
    """Validates and auto-corrects Director JSON output."""
    
    def __init__(self, voice_catalog: List[Dict] = None):
        """
        Initialize validator.
        
        Args:
            voice_catalog: List of available voices with metadata
        """
        if voice_catalog is None:
            # Auto-load voice catalog from knowledge_bases
            import os.path as path
            catalog_path = path.join(
                path.dirname(__file__), 
                'knowledge_bases', 
                'voice_embeddings_store',
                'voices_metadata.json'
            )
            try:
                with open(catalog_path, 'r') as f:
                    self.voice_catalog = json.load(f)
                logger.info(f"Loaded {len(self.voice_catalog)} voices for validation")
            except FileNotFoundError:
                logger.warning(f"Voice catalog not found at {catalog_path}")
                self.voice_catalog = []
            except Exception as e:
                logger.error(f"Error loading voice catalog: {e}")
                self.voice_catalog = []
        else:
            self.voice_catalog = voice_catalog
        
        self.errors = []
    
    def validate(self, director_json: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Validate Director JSON.
        
        Returns:
            (is_valid, errors_list)
        """
        self.errors = []
        
        logger.info("=== STARTING DIRECTOR JSON VALIDATION ===")
        logger.info(f"Voice catalog size: {len(self.voice_catalog)}")
        logger.info(f"Scenes to validate: {len(director_json.get('scenes', []))}")
        
        # Extract character metadata from dialogue
        characters = self._extract_character_info(director_json)
        
        logger.info(f"Characters detected: {list(characters.keys())}")
        for char_name, info in characters.items():
            logger.info(f"  {char_name}: age={info.get('age')}, ethnicity={info.get('ethnicity')}, voice_id={info.get('voice_key')}")
        
        # Validate voice assignments
        self._validate_voice_assignments(director_json, characters)
        
        # Validate voice uniqueness
        self._validate_voice_uniqueness(director_json)
        
        logger.info(f"Validation complete: {len(self.errors)} errors found")
        
        return len(self.errors) == 0, self.errors
    
    def auto_correct(self, director_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically fix validation errors.
        
        Returns:
            Corrected Director JSON
        """
        # First validate to find errors
        is_valid, errors = self.validate(director_json)
        
        if is_valid:
            logger.info("No validation errors found - no corrections needed")
            return director_json
        
        logger.warning(f"Found {len(errors)} validation errors - applying auto-corrections")
        
        corrected_json = director_json.copy()
        
        for error in errors:
            if error.error_type == "TEEN_VOICE_WRONG_LABEL":
                corrected_json = self._fix_teen_voice(corrected_json, error)
            
            elif error.error_type == "ETHNICITY_MISMATCH":
                corrected_json = self._fix_ethnicity_mismatch(corrected_json, error)
            
            elif error.error_type == "DUPLICATE_VOICE":
                corrected_json = self._fix_duplicate_voice(corrected_json, error)
        
        # Validate again to confirm fixes
        is_valid_after, remaining_errors = self.validate(corrected_json)
        
        if remaining_errors:
            logger.warning(f"Auto-correction incomplete - {len(remaining_errors)} errors remain")
        else:
            logger.info("All validation errors successfully corrected")
        
        return corrected_json
    
    def _extract_character_info(self, director_json: Dict) -> Dict[str, Dict]:
        """Extract character names, ages, ethnicities from Director JSON metadata fields."""
        characters = {}
        
        # Extract character info from dialogue lines with metadata
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                char_name = line.get('character')
                
                if char_name and char_name != 'Narrator':
                    if char_name not in characters:
                        # Initialize character entry
                        characters[char_name] = {
                            'name': char_name,
                            'age': None,
                            'ethnicity': None,
                            'gender': None,
                            'voice_key': line.get('voice_key')
                        }
                    
                    # Extract metadata from this line if present
                    # The AI Director includes these fields on the character's FIRST dialogue line
                    if 'character_age' in line:
                        characters[char_name]['age'] = line['character_age']
                        logger.info(f"✓ Found age metadata: {char_name} = {line['character_age']} years old")
                    
                    if 'character_gender' in line:
                        characters[char_name]['gender'] = line['character_gender']
                        logger.info(f"✓ Found gender metadata: {char_name} = {line['character_gender']}")
                    
                    if 'character_ethnicity' in line:
                        characters[char_name]['ethnicity'] = line['character_ethnicity']
                        logger.info(f"✓ Found ethnicity metadata: {char_name} = {line['character_ethnicity']}")
        
        return characters
    
    def _extract_character_attributes(self, text: str, characters: Dict):
        """Extract age and ethnicity from descriptive text."""
        for char_name in characters.keys():
            # Look for age mentions near character name
            # Patterns: "Derek (17)" "Derek, 17," "17-year-old Derek" "Derek, a 19 year old"
            age_patterns = [
                rf'{char_name}\s*\((\d+)\)',  # Derek (17)
                rf'{char_name},?\s*(\d+)[,\s]',  # Derek, 17,
                rf'(\d+)[-\s]year[-\s]old\s+{char_name}',  # 17-year-old Derek
                rf'{char_name},?\s*a\s*(\d+)\s*year[-\s]?old',  # Derek, a 17 year old
            ]
            
            for pattern in age_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    age = int(match.group(1))
                    characters[char_name]['age'] = age
                    logger.info(f"Extracted age {age} for {char_name} from text: {text[:100]}")
                    break
            
            # Look for ethnicity mentions
            if 'African American' in text and char_name in text:
                characters[char_name]['ethnicity'] = 'African American'
                logger.info(f"Extracted ethnicity 'African American' for {char_name}")
    
    def _validate_voice_assignments(self, director_json: Dict, characters: Dict):
        """Validate that voice assignments match character attributes."""
        
        # Create voice lookup
        voice_metadata = {v['voice_id']: v for v in self.voice_catalog}
        
        for char_name, char_info in characters.items():
            voice_key = char_info.get('voice_key')
            
            if not voice_key:
                logger.warning(f"No voice assigned to {char_name}")
                continue
            
            if voice_key not in voice_metadata:
                logger.warning(f"Voice {voice_key} not found in catalog for {char_name}")
                continue
            
            voice = voice_metadata[voice_key]
            voice_label = voice.get('label', '')
            voice_style = voice.get('style', '')
            
            # Check teen age validation (17-19 years old)
            char_age = char_info.get('age')
            if char_age and 17 <= char_age <= 19:
                # Teen character MUST have "Teen" in voice label
                if 'Teen' not in voice_label:
                    logger.error(f"VALIDATION ERROR: {char_name} is {char_age} years old but has voice '{voice_label}' (not Teen)")
                    self.errors.append(ValidationError(
                        error_type="TEEN_VOICE_WRONG_LABEL",
                        message=f"Character '{char_name}' is {char_age} years old but has voice label '{voice_label}' instead of 'Teen'",
                        location=f"character:{char_name}",
                        fix_data={
                            'character': char_name,
                            'age': char_age,
                            'current_voice': voice_key,
                            'current_label': voice_label
                        }
                    ))
            
            # Check ethnicity matching
            char_ethnicity = char_info.get('ethnicity')
            if char_ethnicity == 'African American':
                if voice_style != 'African American':
                    logger.error(f"VALIDATION ERROR: {char_name} is African American but has voice style '{voice_style}'")
                    self.errors.append(ValidationError(
                        error_type="ETHNICITY_MISMATCH",
                        message=f"Character '{char_name}' is African American but voice style is '{voice_style}'",
                        location=f"character:{char_name}",
                        fix_data={
                            'character': char_name,
                            'expected_style': 'African American',
                            'current_voice': voice_key,
                            'current_style': voice_style
                        }
                    ))
    
    def _validate_voice_uniqueness(self, director_json: Dict):
        """Validate that each character has a unique voice."""
        voice_usage = {}
        
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                char_name = line.get('character')
                voice_key = line.get('voice_key')
                
                if char_name == 'Narrator':
                    continue
                
                if voice_key not in voice_usage:
                    voice_usage[voice_key] = []
                
                if char_name not in voice_usage[voice_key]:
                    voice_usage[voice_key].append(char_name)
        
        # Find duplicates
        for voice_key, characters in voice_usage.items():
            if len(characters) > 1:
                logger.error(f"VALIDATION ERROR: Voice {voice_key} assigned to multiple characters: {characters}")
                self.errors.append(ValidationError(
                    error_type="DUPLICATE_VOICE",
                    message=f"Voice {voice_key} assigned to multiple characters: {characters}",
                    location=f"voice:{voice_key}",
                    fix_data={'voice_key': voice_key, 'characters': characters}
                ))
    
    def _fix_teen_voice(self, director_json: Dict, error: ValidationError) -> Dict:
        """Fix teen character with wrong voice label."""
        char_name = error.fix_data['character']
        char_age = error.fix_data.get('age')
        current_voice = error.fix_data.get('current_voice')
        
        logger.info(f"Attempting to fix teen voice for {char_name} (age {char_age}, current voice: {current_voice})")
        
        # Find teen voices from catalog
        # Get current voice to match gender if possible
        current_voice_data = next((v for v in self.voice_catalog if v['voice_id'] == current_voice), None)
        current_gender = current_voice_data.get('gender') if current_voice_data else None
        
        # Find teen voices, prefer same gender
        teen_voices = [
            v for v in self.voice_catalog 
            if 'Teen' in v.get('label', '')
        ]
        
        if not teen_voices:
            logger.warning(f"No teen voices available in catalog to fix {char_name}")
            return director_json
        
        # Try to match gender if known
        if current_gender:
            matching_gender = [v for v in teen_voices if v.get('gender') == current_gender]
            if matching_gender:
                teen_voices = matching_gender
        
        # Get already used voices to avoid duplicates
        used_voices = set()
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                if line.get('voice_key'):
                    used_voices.add(line['voice_key'])
        
        # Find unused teen voice
        new_voice = None
        for voice in teen_voices:
            if voice['voice_id'] not in used_voices or voice['voice_id'] == current_voice:
                new_voice = voice['voice_id']
                break
        
        if not new_voice:
            # If all used, just take first teen voice
            new_voice = teen_voices[0]['voice_id']
        
        # Update all instances of this character
        updated_count = 0
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                if line.get('character') == char_name:
                    line['voice_key'] = new_voice
                    updated_count += 1
        
        logger.info(f"✓ Fixed {char_name}: Changed voice from {current_voice} to {new_voice} ({updated_count} dialogue lines updated)")
        return director_json
    
    def _fix_ethnicity_mismatch(self, director_json: Dict, error: ValidationError) -> Dict:
        """Fix ethnicity mismatch between character and voice."""
        char_name = error.fix_data['character']
        expected_style = error.fix_data['expected_style']
        current_voice = error.fix_data.get('current_voice')
        
        logger.info(f"Attempting to fix ethnicity for {char_name} (needs {expected_style})")
        
        # Find voices matching ethnicity
        matching_voices = [
            v for v in self.voice_catalog 
            if v.get('style') == expected_style
        ]
        
        if not matching_voices:
            logger.warning(f"No {expected_style} voices available to fix {char_name}")
            return director_json
        
        # Get used voices
        used_voices = set()
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                if line.get('voice_key'):
                    used_voices.add(line['voice_key'])
        
        # Find unused matching voice
        new_voice = None
        for voice in matching_voices:
            if voice['voice_id'] not in used_voices:
                new_voice = voice['voice_id']
                break
        
        if not new_voice:
            new_voice = matching_voices[0]['voice_id']
        
        # Update all instances
        updated_count = 0
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                if line.get('character') == char_name:
                    line['voice_key'] = new_voice
                    updated_count += 1
        
        logger.info(f"✓ Fixed {char_name}: Changed to {expected_style} voice {new_voice} ({updated_count} lines updated)")
        return director_json
    
    def _fix_duplicate_voice(self, director_json: Dict, error: ValidationError) -> Dict:
        """Fix duplicate voice assignments."""
        voice_key = error.fix_data['voice_key']
        characters = error.fix_data['characters']
        
        logger.info(f"Fixing duplicate voice {voice_key} used by: {characters}")
        
        # Keep first character with this voice, reassign others
        for char_name in characters[1:]:
            # Find alternative voice
            alternative = self._find_alternative_voice(voice_key, director_json)
            
            if alternative:
                # Update all instances
                updated_count = 0
                for scene in director_json.get('scenes', []):
                    for line in scene.get('dialogue', []):
                        if line.get('character') == char_name:
                            line['voice_key'] = alternative
                            updated_count += 1
                
                logger.info(f"✓ Fixed duplicate: {char_name} reassigned from {voice_key} to {alternative} ({updated_count} lines)")
        
        return director_json
    
    def _find_alternative_voice(self, current_voice: str, director_json: Dict) -> str:
        """Find an alternative voice not currently in use."""
        # Get all currently used voices
        used_voices = set()
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                if line.get('voice_key'):
                    used_voices.add(line['voice_key'])
        
        # Find unused voice from catalog
        for voice in self.voice_catalog:
            if voice['voice_id'] not in used_voices:
                return voice['voice_id']
        
        # If all voices used, return first available (fallback)
        return self.voice_catalog[0]['voice_id'] if self.voice_catalog else current_voice


class SceneAudioValidator:
    """Validates and auto-corrects Scene Audio JSON output."""
    
    def __init__(self):
        self.errors = []
    
    def validate(self, scene_audio_json: Dict[str, Any], director_json: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Validate Scene Audio JSON.
        
        Args:
            scene_audio_json: Audio selections for scenes
            director_json: Original Director JSON for context
        
        Returns:
            (is_valid, errors_list)
        """
        self.errors = []
        
        # Validate ambience matches scene headers
        self._validate_ambience(scene_audio_json, director_json)
        
        # Validate SFX only on explicit actions
        self._validate_sfx(scene_audio_json, director_json)
        
        return len(self.errors) == 0, self.errors
    
    def auto_correct(self, scene_audio_json: Dict[str, Any], director_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically fix validation errors.
        
        Returns:
            Corrected Scene Audio JSON
        """
        is_valid, errors = self.validate(scene_audio_json, director_json)
        
        if is_valid:
            logger.info("Scene audio validation passed - no corrections needed")
            return scene_audio_json
        
        logger.warning(f"Found {len(errors)} scene audio errors - applying corrections")
        
        corrected_json = scene_audio_json.copy()
        
        for error in errors:
            if error.error_type == "AMBIENCE_LOCATION_MISMATCH":
                corrected_json = self._fix_ambience_mismatch(corrected_json, error)
            
            elif error.error_type == "INVALID_SFX":
                corrected_json = self._remove_invalid_sfx(corrected_json, error)
        
        return corrected_json
    
    def _validate_ambience(self, scene_audio_json: Dict, director_json: Dict):
        """Validate ambience matches scene location (INT/EXT)."""
        # Implementation for ambience validation
        pass
    
    def _validate_sfx(self, scene_audio_json: Dict, director_json: Dict):
        """Validate SFX only on capitalized/explicit actions."""
        # Implementation for SFX validation
        pass
    
    def _fix_ambience_mismatch(self, scene_audio_json: Dict, error: ValidationError) -> Dict:
        """Fix ambience location mismatches."""
        logger.info(f"Fixing ambience mismatch: {error.message}")
        return scene_audio_json
    
    def _remove_invalid_sfx(self, scene_audio_json: Dict, error: ValidationError) -> Dict:
        """Remove SFX that shouldn't be there."""
        logger.info(f"Removing invalid SFX: {error.message}")
        return scene_audio_json
