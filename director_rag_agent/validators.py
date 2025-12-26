"""
Validation and Auto-Correction Layer

Validates agent outputs and automatically corrects common mistakes
to ensure consistency and reliability before audio generation.
"""

import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional

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
        
        # Load knowledge base for validation rules
        self.knowledge = AUDIO_KNOWLEDGE
        
        # Extract location patterns from ambience examples
        self.location_patterns = self._build_location_patterns()
    
    def _build_location_patterns(self) -> Dict[str, str]:
        """Build location keyword to ambience file mapping from knowledge base."""
        patterns = {}
        
        for example in self.knowledge.get('ambience_examples', []):
            header = example['scene_header'].upper()
            ambience = example['ambience_file']
            
            # Extract key location words
            if 'FOREST' in header or 'WOODS' in header:
                patterns['FOREST'] = 'a_forest_calm_01.mp3'
                patterns['WOODS'] = 'a_forest_calm_01.mp3'
            elif 'PARKING LOT' in header:
                patterns['PARKING LOT'] = 'a_parking_lot_small_calm_crowds_01'
            elif 'FRONT YARD' in header and 'COUNTRY' in header:
                patterns['COUNTRY'] = 'a_countryside_birds_01'
            elif 'CAFETERIA' in header:
                patterns['CAFETERIA'] = 'a_public_spaces_people_mall_01'
            elif 'ROOFTOP POOL' in header:
                patterns['ROOFTOP POOL'] = 'a_city_streets_small_calm_crowds_01'
            elif 'HOSPITAL' in header:
                patterns['HOSPITAL'] = 'a_hospital_hallway_01'
            elif 'MEDICAL TENT' in header or 'ARMY BASE' in header:
                patterns['MILITARY'] = 'a_military_medical_tent_01'
        
        return patterns
    
    def validate(self, scene_audio_json: Dict[str, Any], director_json: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Validate Scene Audio JSON against knowledge base rules.
        
        Args:
            scene_audio_json: Audio selections for scenes
            director_json: Original Director JSON for context
        
        Returns:
            (is_valid, errors_list)
        """
        self.errors = []
        
        logger.info("=== STARTING SCENE AUDIO VALIDATION ===")
        
        # Validate each scene's audio selections
        for scene_audio in scene_audio_json.get('scenes', []):
            scene_id = scene_audio.get('scene_id')
            
            # Find corresponding scene in director JSON
            director_scene = self._find_director_scene(director_json, scene_id)
            if not director_scene:
                logger.warning(f"Could not find director scene for {scene_id}")
                continue
            
            logger.info(f"Validating audio for {scene_id}")
            
            # Validate music
            self._validate_music(scene_audio, director_scene, scene_id)
            
            # Validate ambience
            self._validate_ambience(scene_audio, director_scene, scene_id)
            
            # Validate SFX
            self._validate_sfx(scene_audio, director_scene, scene_id)
        
        logger.info(f"Scene audio validation complete: {len(self.errors)} errors found")
        
        return len(self.errors) == 0, self.errors
    
    def _find_director_scene(self, director_json: Dict, scene_id: str) -> Optional[Dict]:
        """Find the corresponding scene in Director JSON."""
        for scene in director_json.get('scenes', []):
            if scene.get('scene_id') == scene_id:
                return scene
        return None
    
    def _validate_music(self, scene_audio: Dict, director_scene: Dict, scene_id: str):
        """Validate music selection for scene."""
        music = scene_audio.get('music', [])
        
        # RULE 1: Every scene should have music
        if not music or len(music) == 0:
            logger.warning(f"No music selected for {scene_id}")
            self.errors.append(ValidationError(
                error_type="MISSING_MUSIC",
                message=f"Scene {scene_id} has no music - every scene should have background music",
                location=scene_id,
                fix_data={'scene_id': scene_id, 'director_scene': director_scene}
            ))
            return
        
        # RULE 2: Check music file exists and mood is reasonable
        music_file = music[0].get('file', '')
        
        # Extract scene mood from director JSON
        scene_mood = director_scene.get('mood', '').lower()
        scene_intensity = director_scene.get('intensity', 0.5)
        
        # Check for mood contradictions based on keywords
        mood_keywords = {
            'tense': ['tense', 'dramatic', 'scary', 'mysterious'],
            'sad': ['sad', 'tragic', 'emotional', 'grief'],
            'happy': ['uplifting', 'love', 'playful', 'groovy'],
            'romantic': ['love', 'romantic'],
            'scary': ['scary', 'mysterious', 'horror']
        }
        
        # Check if music file mood contradicts scene mood
        for scene_keyword, valid_music_moods in mood_keywords.items():
            if scene_keyword in scene_mood:
                # Check if music file contains any valid mood keyword
                has_valid_mood = any(mood in music_file.lower() for mood in valid_music_moods)
                
                if not has_valid_mood:
                    # Check for contradictions
                    if scene_keyword == 'sad' and ('uplifting' in music_file.lower() or 'groovy' in music_file.lower()):
                        logger.error(f"Music mood contradiction in {scene_id}: sad scene has {music_file}")
                        self.errors.append(ValidationError(
                            error_type="MUSIC_MOOD_MISMATCH",
                            message=f"Sad scene has inappropriate music: {music_file}",
                            location=scene_id,
                            fix_data={'scene_id': scene_id, 'scene_mood': scene_mood, 'current_music': music_file}
                        ))
                    elif scene_keyword == 'tense' and ('playful' in music_file.lower() or 'groovy' in music_file.lower()):
                        logger.error(f"Music mood contradiction in {scene_id}: tense scene has {music_file}")
                        self.errors.append(ValidationError(
                            error_type="MUSIC_MOOD_MISMATCH",
                            message=f"Tense scene has inappropriate music: {music_file}",
                            location=scene_id,
                            fix_data={'scene_id': scene_id, 'scene_mood': scene_mood, 'current_music': music_file}
                        ))
        
        logger.info(f"✓ Music validation passed for {scene_id}")
    
    def _validate_ambience(self, scene_audio: Dict, director_scene: Dict, scene_id: str):
        """Validate ambience selection matches scene header location ONLY."""
        ambience = scene_audio.get('ambience', [])
        
        # Get scene header from first narrator line
        scene_header = self._extract_scene_header(director_scene)
        
        if not scene_header:
            logger.warning(f"Could not extract scene header for {scene_id}")
            return
        
        logger.info(f"Scene header: {scene_header}")
        
        # Determine if INT or EXT
        is_interior = scene_header.upper().startswith('INT')
        is_exterior = scene_header.upper().startswith('EXT')
        
        # Check if location matches any known patterns that should have ambience
        should_have_ambience = False
        expected_ambience_type = None
        
        header_upper = scene_header.upper()
        for keyword, ambience_file in self.location_patterns.items():
            if keyword in header_upper:
                should_have_ambience = True
                expected_ambience_type = keyword
                break
        
        # RULE 1: Scenes with matchable locations should have ambience
        if should_have_ambience and (not ambience or len(ambience) == 0):
            logger.error(f"Missing ambience for {scene_id}: Header '{scene_header}' should have {expected_ambience_type} ambience")
            self.errors.append(ValidationError(
                error_type="MISSING_AMBIENCE",
                message=f"Scene header '{scene_header}' should have ambience for {expected_ambience_type}",
                location=scene_id,
                fix_data={
                    'scene_id': scene_id,
                    'scene_header': scene_header,
                    'expected_type': expected_ambience_type,
                    'expected_file': self.location_patterns.get(expected_ambience_type)
                }
            ))
            return
        
        # If ambience exists, validate it matches scene header
        if ambience and len(ambience) > 0:
            ambience_file = ambience[0].get('file', '')
            
            # RULE 2: INT scenes should NOT have outdoor ambience
            outdoor_keywords = ['forest', 'street', 'city', 'park', 'countryside', 'ocean', 'beach']
            indoor_keywords = ['hospital', 'cafe', 'office', 'mall', 'hallway']
            
            if is_interior:
                has_outdoor_ambience = any(keyword in ambience_file.lower() for keyword in outdoor_keywords)
                if has_outdoor_ambience:
                    logger.error(f"INT/EXT mismatch in {scene_id}: Interior scene has outdoor ambience {ambience_file}")
                    self.errors.append(ValidationError(
                        error_type="AMBIENCE_INT_EXT_MISMATCH",
                        message=f"Interior scene '{scene_header}' has outdoor ambience: {ambience_file}",
                        location=scene_id,
                        fix_data={'scene_id': scene_id, 'scene_header': scene_header, 'wrong_ambience': ambience_file}
                    ))
            
            elif is_exterior:
                has_indoor_ambience = any(keyword in ambience_file.lower() for keyword in indoor_keywords)
                if has_indoor_ambience:
                    logger.error(f"INT/EXT mismatch in {scene_id}: Exterior scene has indoor ambience {ambience_file}")
                    self.errors.append(ValidationError(
                        error_type="AMBIENCE_INT_EXT_MISMATCH",
                        message=f"Exterior scene '{scene_header}' has indoor ambience: {ambience_file}",
                        location=scene_id,
                        fix_data={'scene_id': scene_id, 'scene_header': scene_header, 'wrong_ambience': ambience_file}
                    ))
        
        if not self.errors or scene_id not in [e.location for e in self.errors]:
            logger.info(f"✓ Ambience validation passed for {scene_id}")
    
    def _validate_sfx(self, scene_audio: Dict, director_scene: Dict, scene_id: str):
        """Validate SFX only on explicit/capitalized actions."""
        sfx_list = scene_audio.get('sfx', [])
        
        if not sfx_list:
            logger.info(f"No SFX in {scene_id} - acceptable if no explicit actions")
            return
        
        logger.info(f"Validating {len(sfx_list)} SFX in {scene_id}")
        
        # Get all dialogue/action text to search for trigger words
        all_text = []
        line_id_to_text = {}
        
        for line in director_scene.get('dialogue', []):
            line_id = line.get('line_id')
            text = line.get('text', '')
            all_text.append(text)
            line_id_to_text[line_id] = text
        
        combined_text = ' '.join(all_text)
        
        # Check each SFX
        for sfx in sfx_list:
            sfx_file = sfx.get('file', '')
            start_at = sfx.get('start_at')
            
            # Try to find trigger text for this SFX
            trigger_text = line_id_to_text.get(start_at, '')
            
            # Check for explicit trigger words from knowledge base
            has_explicit_trigger = self._check_sfx_trigger(sfx_file, trigger_text, combined_text)
            
            if not has_explicit_trigger:
                logger.warning(f"SFX in {scene_id} may not have explicit trigger: {sfx_file} at {start_at}")
                self.errors.append(ValidationError(
                    error_type="SFX_NO_EXPLICIT_TRIGGER",
                    message=f"SFX {sfx_file} may not have explicit capitalized trigger in script",
                    location=f"{scene_id}:{start_at}",
                    fix_data={'scene_id': scene_id, 'sfx_file': sfx_file, 'line_id': start_at}
                ))
        
        if not any(e.error_type == "SFX_NO_EXPLICIT_TRIGGER" and scene_id in e.location for e in self.errors):
            logger.info(f"✓ SFX validation passed for {scene_id}")
    
    def _check_sfx_trigger(self, sfx_file: str, trigger_text: str, full_text: str) -> bool:
        """Check if SFX has explicit trigger word in text."""
        # Extract SFX type from filename
        sfx_type = sfx_file.lower()
        
        # Define trigger words for common SFX types
        trigger_patterns = {
            'door': ['slam', 'slams', 'shut', 'opens', 'knock', 'knocks'],
            'water': ['splash', 'crash', 'crashes', 'pour', 'rushing'],
            'punch': ['punch', 'punches', 'hit', 'hits', 'slam'],
            'fall': ['fall', 'falls', 'slam', 'crash', 'impact'],
            'gun': ['shot', 'shoot', 'fire', 'gunshot'],
            'car': ['door closes', 'engine', 'accelerate', 'screech']
        }
        
        # Check if SFX type matches any trigger pattern
        for sfx_keyword, triggers in trigger_patterns.items():
            if sfx_keyword in sfx_type:
                # Check if any trigger word appears in text (preferably capitalized)
                for trigger in triggers:
                    if trigger.upper() in full_text.upper():
                        return True
                    if trigger in trigger_text.lower():
                        return True
        
        # If no specific pattern match, check for general capitalization emphasis
        words = trigger_text.split()
        has_caps = any(word.isupper() and len(word) > 2 for word in words)
        
        return has_caps
    
    def _extract_scene_header(self, director_scene: Dict) -> str:
        """Extract scene header from first narrator line."""
        dialogue = director_scene.get('dialogue', [])
        
        for line in dialogue:
            if line.get('character') == 'Narrator':
                text = line.get('text', '')
                # Scene headers typically have INT/EXT
                if text.upper().startswith('INT') or text.upper().startswith('EXT') or text.upper().startswith('I.E'):
                    return text
        
        return ""
    
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
        
        corrected_json = json.loads(json.dumps(scene_audio_json))  # Deep copy
        
        for error in errors:
            if error.error_type == "MISSING_AMBIENCE":
                corrected_json = self._fix_missing_ambience(corrected_json, error)
            
            elif error.error_type == "AMBIENCE_INT_EXT_MISMATCH":
                corrected_json = self._fix_ambience_mismatch(corrected_json, error)
            
            elif error.error_type == "SFX_NO_EXPLICIT_TRIGGER":
                corrected_json = self._remove_invalid_sfx(corrected_json, error)
            
            elif error.error_type == "MISSING_MUSIC":
                corrected_json = self._add_default_music(corrected_json, error)
            
            elif error.error_type == "MUSIC_MOOD_MISMATCH":
                corrected_json = self._fix_music_mood(corrected_json, error)
        
        logger.info(f"Scene audio corrections applied")
        
        return corrected_json
    
    def _fix_missing_ambience(self, scene_audio_json: Dict, error: ValidationError) -> Dict:
        """Add missing ambience based on scene header."""
        scene_id = error.fix_data['scene_id']
        expected_file = error.fix_data.get('expected_file')
        
        if not expected_file:
            logger.warning(f"Cannot fix missing ambience for {scene_id} - no expected file")
            return scene_audio_json
        
        # Find the scene in audio JSON
        for scene in scene_audio_json.get('scenes', []):
            if scene.get('scene_id') == scene_id:
                # Add ambience
                if 'ambience' not in scene:
                    scene['ambience'] = []
                
                scene['ambience'] = [{'file': expected_file}]
                logger.info(f"✓ Added missing ambience to {scene_id}: {expected_file}")
                break
        
        return scene_audio_json
    
    def _fix_ambience_mismatch(self, scene_audio_json: Dict, error: ValidationError) -> Dict:
        """Remove incorrect ambience."""
        scene_id = error.fix_data['scene_id']
        
        # Find and remove incorrect ambience
        for scene in scene_audio_json.get('scenes', []):
            if scene.get('scene_id') == scene_id:
                scene['ambience'] = []
                logger.info(f"✓ Removed incorrect ambience from {scene_id}")
                break
        
        return scene_audio_json
    
    def _remove_invalid_sfx(self, scene_audio_json: Dict, error: ValidationError) -> Dict:
        """Remove SFX without explicit triggers."""
        scene_id = error.fix_data['scene_id']
        sfx_file = error.fix_data['sfx_file']
        
        for scene in scene_audio_json.get('scenes', []):
            if scene.get('scene_id') == scene_id:
                # Remove the problematic SFX
                scene['sfx'] = [s for s in scene.get('sfx', []) if s.get('file') != sfx_file]
                logger.info(f"✓ Removed invalid SFX from {scene_id}: {sfx_file}")
                break
        
        return scene_audio_json
    
    def _add_default_music(self, scene_audio_json: Dict, error: ValidationError) -> Dict:
        """Add default music to scene without music."""
        scene_id = error.fix_data['scene_id']
        director_scene = error.fix_data['director_scene']
        
        # Determine appropriate default music based on scene mood
        scene_mood = director_scene.get('mood', '').lower()
        
        default_music = 'm_piano_strings_dramatic_01.mp3'  # Safe default
        
        if 'tense' in scene_mood or 'conflict' in scene_mood:
            default_music = 'm_strings_orchestra_tense_01.mp3'
        elif 'sad' in scene_mood or 'emotional' in scene_mood:
            default_music = 'm_piano_slow_sad_01.mp3'
        elif 'happy' in scene_mood or 'uplifting' in scene_mood:
            default_music = 'm_string_orchestra_uplifting_01.mp3'
        
        for scene in scene_audio_json.get('scenes', []):
            if scene.get('scene_id') == scene_id:
                scene['music'] = [{'file': f'music/{default_music}'}]
                logger.info(f"✓ Added default music to {scene_id}: {default_music}")
                break
        
        return scene_audio_json
    
    def _fix_music_mood(self, scene_audio_json: Dict, error: ValidationError) -> Dict:
        """Fix music mood mismatch."""
        scene_id = error.fix_data['scene_id']
        scene_mood = error.fix_data['scene_mood']
        
        # Select appropriate music based on scene mood
        replacement_music = 'm_piano_strings_dramatic_01.mp3'
        
        if 'sad' in scene_mood:
            replacement_music = 'm_piano_slow_sad_01.mp3'
        elif 'tense' in scene_mood:
            replacement_music = 'm_strings_orchestra_tense_01.mp3'
        
        for scene in scene_audio_json.get('scenes', []):
            if scene.get('scene_id') == scene_id:
                scene['music'] = [{'file': f'music/{replacement_music}'}]
                logger.info(f"✓ Fixed music mood for {scene_id}: {replacement_music}")
                break
        
        return scene_audio_json
