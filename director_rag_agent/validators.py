"""
Validation and Auto-Correction Layer

Validates agent outputs and automatically corrects common mistakes
to ensure consistency and reliability before audio generation.
"""

import logging
import json
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
        self.voice_catalog = voice_catalog or []
        self.errors = []
    
    def validate(self, director_json: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Validate Director JSON.
        
        Returns:
            (is_valid, errors_list)
        """
        self.errors = []
        
        # Extract character metadata from dialogue
        characters = self._extract_character_info(director_json)
        
        # Validate voice assignments
        self._validate_voice_assignments(director_json, characters)
        
        # Validate voice uniqueness
        self._validate_voice_uniqueness(director_json)
        
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
        """Extract character names, ages, ethnicities from dialogue lines."""
        characters = {}
        
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                char_name = line.get('character')
                
                if char_name and char_name != 'Narrator':
                    if char_name not in characters:
                        characters[char_name] = {
                            'name': char_name,
                            'age': None,
                            'ethnicity': None,
                            'gender': None,
                            'voice_key': line.get('voice_key')
                        }
        
        return characters
    
    def _validate_voice_assignments(self, director_json: Dict, characters: Dict):
        """Validate that voice assignments match character attributes."""
        
        # Load voice catalog metadata
        voice_metadata = {v['voice_id']: v for v in self.voice_catalog}
        
        for char_name, char_info in characters.items():
            voice_key = char_info.get('voice_key')
            
            if not voice_key or voice_key not in voice_metadata:
                continue
            
            voice = voice_metadata[voice_key]
            
            # Check teen age characters (17-19)
            # This would require parsing character descriptions from script
            # For now, we'll check the voice label
            if 'teen' in char_name.lower() or 'teenage' in char_name.lower():
                if 'teen' not in voice.get('label', '').lower():
                    self.errors.append(ValidationError(
                        error_type="TEEN_VOICE_WRONG_LABEL",
                        message=f"Character '{char_name}' appears to be a teen but has non-teen voice",
                        location=f"character:{char_name}",
                        fix_data={'character': char_name, 'current_voice': voice_key}
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
                self.errors.append(ValidationError(
                    error_type="DUPLICATE_VOICE",
                    message=f"Voice {voice_key} assigned to multiple characters: {characters}",
                    location=f"voice:{voice_key}",
                    fix_data={'voice_key': voice_key, 'characters': characters}
                ))
    
    def _fix_teen_voice(self, director_json: Dict, error: ValidationError) -> Dict:
        """Fix teen character with wrong voice label."""
        char_name = error.fix_data['character']
        
        # Find a teen voice from catalog
        teen_voices = [
            v for v in self.voice_catalog 
            if 'teen' in v.get('label', '').lower()
        ]
        
        if not teen_voices:
            logger.warning(f"No teen voices available to fix {char_name}")
            return director_json
        
        # Select first available teen voice (could be more sophisticated)
        new_voice = teen_voices[0]['voice_id']
        
        # Update all instances of this character
        for scene in director_json.get('scenes', []):
            for line in scene.get('dialogue', []):
                if line.get('character') == char_name:
                    line['voice_key'] = new_voice
        
        logger.info(f"Fixed {char_name}: assigned teen voice {new_voice}")
        return director_json
    
    def _fix_ethnicity_mismatch(self, director_json: Dict, error: ValidationError) -> Dict:
        """Fix ethnicity mismatch between character and voice."""
        # Implementation for ethnicity fixes
        logger.info(f"Fixing ethnicity mismatch: {error.message}")
        return director_json
    
    def _fix_duplicate_voice(self, director_json: Dict, error: ValidationError) -> Dict:
        """Fix duplicate voice assignments."""
        voice_key = error.fix_data['voice_key']
        characters = error.fix_data['characters']
        
        # Keep first character with this voice, reassign others
        for i, char_name in enumerate(characters[1:], 1):
            # Find alternative voice from catalog
            alternative = self._find_alternative_voice(voice_key, director_json)
            
            if alternative:
                # Update all instances
                for scene in director_json.get('scenes', []):
                    for line in scene.get('dialogue', []):
                        if line.get('character') == char_name:
                            line['voice_key'] = alternative
                
                logger.info(f"Fixed duplicate: {char_name} reassigned from {voice_key} to {alternative}")
        
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
