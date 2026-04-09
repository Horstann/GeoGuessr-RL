from ai_client import get_openai_client
import re
import os
import base64
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
from geopy.geocoders import Nominatim

class GeoAoTAI:
    """
    Multi-turn conversation system for GeoAoT (Action of Thought) geo-guessing using OpenAI-compatible API.
    Paper: Learning to Wander: Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning
    Maintains conversation history with images and provides structured response parsing.
    """
    
    def __init__(self,
                 ai_config=None,
                 ai_keys=None):
        """Initialize the GeoAoT (Action of Thought) AI conversation system."""
        # Create unified client with direct config
        self.client = get_openai_client(ai_config=ai_config, ai_keys=ai_keys)
        
        # Set parameters from config
        self.model = ai_config.model
        self.max_tokens = ai_config.max_tokens
        self.temperature = ai_config.temperature
        self.final_temp = self.temperature
        self.base_url = ai_config.base_url
        
        # Google API key
        self.google_api_key = ai_keys.google_api_key
        self.conversation_history = []
        self.step_count = 0

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.token_call_records = []  # Per-call token breakdown
        self._call_counter = 0         # Sequential call counter

        # Google Geocoding API configuration (already set above)
        self.geocoding_log_file = "geocoding_results.json"
        self.geocoding_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests to respect rate limits
        
    def _add_to_history(self, role: str, content: str, image_path: str = None, step_info: Dict = None):
        """Add message to conversation history with optional image and step context."""
        message = {
            "role": role, 
            "content": content,
            "timestamp": self._get_timestamp()
        }
        if image_path:
            message["image_path"] = image_path
        if step_info:
            message["step_info"] = step_info
        self.conversation_history.append(message)
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API."""
        if isinstance(image_path, str):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # Handle PIL Image objects
            img_buffer = io.BytesIO()
            image_path.save(img_buffer, format='PNG')
            return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    def _create_conversation_context(self) -> str:
        """Create context summary from conversation history."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for i, msg in enumerate(self.conversation_history[-5:]):  # Last 5 messages
            role = msg["role"]
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            
            if "step_info" in msg:
                step_info = msg["step_info"]
                context_parts.append(f"Step {step_info.get('step', i+1)} ({role}): {content}")
            else:
                context_parts.append(f"{role.capitalize()}: {content}")
        
        return "\\n".join(context_parts)
    
    def get_navigation_decision(self, 
                              composite_image, 
                              current_step: int,
                              current_node: str,
                              current_heading: float,
                              current_rotation: int,
                              available_moves: List[Dict],
                              location_info: str,
                              max_steps: int) -> str:
        """
        Get AI decision for next navigation action in GeoAoT (Action of Thought) exploration.
        Natural conversation approach - AI sees the image and makes decisions naturally.
        """
        self.step_count = current_step
        
        # For the first step, set up the system prompt
        if current_step == 1:
            system_prompt = """IMPORTANT: Always respond in this unified JSON format:
{
  "observations": "One or two short sentence describe you see in the image",
  "confidence": "low" | "medium" | "high",
  "action": "continue" | "guess",
  "next_move": {
    "type": "rotate_degrees" | "move_to_color" | "back_to_original",
    "details": "45" | "red" | null
  },
  "final_guess": {
    "method": "detailed_description" | "coordinates",
    "location_description": "[Address] [Street Name], [City], [State], [Country]",
    "coordinates": {"latitude": 34.0522, "longitude": -118.2437}
  }
}

Actions:
- If action = "continue": Provide next_move (rotate_degrees/move_to_color/back_to_original)
- If action = "guess": Provide final_guess with location

When making final_guess:
- method "detailed_description": MUST use EXACT format "[Address] [Street Name], [City], [State], [Country]" OR reduced specificity like "[City], [State], [Country]" OR "[State], [Country]" OR "[Country]". NO descriptive text, NO explanations, ONLY location names in this format. If absolutely no geographical clues available, use EXACTLY "FAIL TO PREDICT". Set coordinates to null.
- method "coordinates": Set coordinates to {"latitude": X, "longitude": Y}, set location_description to null.

CRITICAL: location_description must be ONLY location names in the specified format. NO descriptive phrases like "suburban area" or "appears to be". 

WARNING: Using "FAIL TO PREDICT" results in severe punishment. When uncertain, strongly consider using coordinates method instead."""
            
            self._add_to_history("system", system_prompt)
        
        # Create simple, natural prompt
        if current_step == 1:
            user_text = f"Here's the starting panorama view (total continue step expect guess {current_step}/{max_steps - 1}). Analyze the image and respond in JSON format."
        else:
            user_text = f"Here's your current view after your last action (total continue step except guess {current_step}/{max_steps - 1}). Analyze this new view and respond in JSON format."
        user_text += f" Aviable move action is {[color['color'] for color in available_moves]}"
        
        # Add current request to conversation
        self._add_to_history("user", user_text, image_path=composite_image, 
                           step_info={
                               "step": current_step,
                               "current_node": current_node,
                               "current_heading": current_heading,
                               "current_rotation": current_rotation,
                               "available_moves": len(available_moves),
                               "max_steps": max_steps
                           })
        
        try:
            # Build conversation messages naturally
            messages = []
            
            for msg in self.conversation_history:
                if msg["role"] == "system":
                    messages.append({"role": "system", "content": msg["content"]})
                elif msg["role"] in ["user", "assistant"]:
                    if "image_path" in msg and msg["role"] == "user":
                        # User message with image
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": msg["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{self._encode_image(msg['image_path'])}"
                                    }
                                }
                            ]
                        })
                    else:
                        # Text-only message
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            ai_response = response.choices[0].message.content

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0

                # Accumulate totals (backward compatibility)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                # Record per-call breakdown
                self._call_counter += 1
                self.token_call_records.append({
                    "call_id": self._call_counter,
                    "call_type": "navigation",
                    "step_number": current_step,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "timestamp": self._get_timestamp(),
                    "model": self.model
                })

            self._add_to_history("assistant", ai_response)

            return ai_response

        except Exception as e:
            error_msg = f"AI Error: {e}"
            self._add_to_history("system", error_msg)
            return error_msg
    
    def get_final_location_guess(self, 
                               composite_image, 
                               exploration_summary: List[str],
                               current_node: str,
                               current_heading: float,
                               current_rotation: int,
                               location_info: str) -> str:
        """
        Get final location coordinates guess after GeoAoT (Action of Thought) exploration.
        Natural conversation - AI has context from exploration and makes final guess.
        """
        # Final guess using unified format
        user_text = f"""Provide your location guess using the same JSON format.

Set action = "guess" and provide your final_guess:
- Use method "detailed_description" with ONE clear sentence containing street names, city, state, country
- OR use method "coordinates" with direct latitude/longitude values

Remember: Keep location_description concise and focused for best geocoding accuracy.
"""
        
        # Add to conversation history naturally
        self._add_to_history("user", user_text, image_path=composite_image)
        
        try:
            # Build messages from full conversation history
            messages = []
            
            for msg in self.conversation_history:
                if msg["role"] == "system":
                    messages.append({"role": "system", "content": msg["content"]})
                elif msg["role"] in ["user", "assistant"]:
                    if "image_path" in msg and msg["role"] == "user":
                        # User message with image
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": msg["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{self._encode_image(msg['image_path'])}"
                                    }
                                }
                            ]
                        })
                    else:
                        # Text-only message
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.final_temp  # Lower temperature for final prediction
            )

            ai_response = response.choices[0].message.content

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0

                # Accumulate totals (backward compatibility)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                # Record per-call breakdown
                self._call_counter += 1
                self.token_call_records.append({
                    "call_id": self._call_counter,
                    "call_type": "final_guess",
                    "step_number": None,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "timestamp": self._get_timestamp(),
                    "model": self.model
                })

            self._add_to_history("assistant", ai_response)

            return ai_response

        except Exception as e:
            error_msg = f"Final guess AI Error: {e}"
            self._add_to_history("system", error_msg)
            return error_msg
    
    def parse_action_response(self, response: str) -> Dict[str, Any]:
        """
        Parse AI response to extract navigation action from JSON format.
        
        Returns:
            Dictionary with action type and parameters
        """
        try:
            # Extract JSON from markdown code blocks if present
            json_text = response.strip()
            if json_text.startswith('```json') and json_text.endswith('```'):
                json_text = json_text[7:-3].strip()  # Remove ```json and ```
            elif json_text.startswith('```') and json_text.endswith('```'):
                json_text = json_text[3:-3].strip()  # Remove ``` and ```
            
            # Try to parse as JSON
            data = json.loads(json_text)
            
            action = data.get('action', '').lower()
            
            if action == 'continue':
                next_move = data.get('next_move', {})
                move_type = next_move.get('type', '').lower()
                move_details = next_move.get('details')
                
                if move_type == 'rotate_degrees':
                    try:
                        degrees = int(move_details) if move_details else 0
                        return {'type': 'rotate', 'degrees': degrees}
                    except (ValueError, TypeError):
                        return {'type': 'unknown', 'response': response}
                        
                elif move_type == 'move_to_color':
                    color = str(move_details).lower() if move_details else None
                    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'pink', 'white', 'black']
                    if color in colors:
                        return {'type': 'move', 'color': color}
                    else:
                        return {'type': 'unknown', 'response': response}
                        
                elif move_type == 'back_to_original':
                    return {'type': 'back'}
                    
                else:
                    return {'type': 'unknown', 'response': response}
                    
            elif action == 'guess':
                return {'type': 'guess', 'response': response}
                
            else:
                # Handle legacy format where action is the specific action type
                if action in ['rotate_degrees', 'move_to_color', 'back_to_original']:
                    # Legacy format compatibility - treat action as the direct command
                    if action == 'rotate_degrees':
                        # Try multiple sources for the rotation value
                        details = (data.get('action_details') or 
                                 data.get('details') or
                                 (data.get('next_move', {}).get('details')) or '0')
                        try:
                            degrees = int(details)
                            return {'type': 'rotate', 'degrees': degrees}
                        except (ValueError, TypeError):
                            return {'type': 'unknown', 'response': response}
                    elif action == 'move_to_color':
                        color = (data.get('action_details') or 
                               data.get('details') or
                               (data.get('next_move', {}).get('details')) or '').lower()
                        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'pink', 'white', 'black']
                        if color in colors:
                            return {'type': 'move', 'color': color}
                    elif action == 'back_to_original':
                        return {'type': 'back'}
                            
                return {'type': 'unknown', 'response': response}
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback to text-based parsing for backwards compatibility
            response_clean = response.lower().strip()
            
            # Parse early guess/stop commands - focus on INTENT, not content
            early_guess_phrases = [
                'guess location', 'make a guess', 'my guess', 'location guess', 'final guess',
                'i\'m confident', 'i\'m ready to guess', 'ready to guess', 'confident enough',
                'stop exploring', 'finish exploring', 'done exploring', 'no need to explore',
                'i know where this is', 'i can tell this is', 'clearly this is',
                'obviously this is', 'definitely this is', 'this must be'
            ]
            
            if any(phrase in response_clean for phrase in early_guess_phrases):
                return {'type': 'guess', 'response': response}
            
            # Parse move command by color
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'pink', 'white', 'black']
            for color in colors:
                if color in response_clean:
                    return {'type': 'move', 'color': color}
            
            # Parse rotation
            rotate_match = re.search(r'rotate\s+([-+]?\d+)', response_clean)
            if rotate_match:
                degrees = int(rotate_match.group(1))
                return {'type': 'rotate', 'degrees': degrees}
            
            return {'type': 'unknown', 'response': response}
    
    def parse_coordinates_from_response(self, response: str) -> Optional[Tuple[float, float]]:
        """
        Parse coordinates from AI response text.
        
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        # Look for "Coordinates: lat, lon" format
        pattern = r"Coordinates:\s*([-+]?\d{1,3}(?:\.\d+)?)\s*,\s*([-+]?\d{1,3}(?:\.\d+)?)"
        match = re.search(pattern, response)
        
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                # Basic validation for coordinate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except (ValueError, IndexError):
                pass
        
        # Fallback for less specific format
        pattern_fallback = r"([-+]?\d{1,3}(?:\.\d+)?)\s*,\s*([-+]?\d{1,3}(?:\.\d+)?)"
        match_fallback = re.search(pattern_fallback, response)
        if match_fallback:
            try:
                lat = float(match_fallback.group(1))
                lon = float(match_fallback.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except (ValueError, IndexError):
                pass

        return None
    
    def geocode_location_google(self, location_description: str) -> Dict[str, Any]:
        """
        Converts a location description to coordinates using Google Geocoding API.
        Always returns a dict with results and metadata for inclusion in final output.
        
        Returns:
            Dict containing coordinates, errors, metadata, and full API response
        """
        result = {
            "input_location": location_description,
            "method": "google_geocoding_api",
            "timestamp": time.time(),
            "success": False,
            "coordinates": None,
            "formatted_address": None,
            "error": None,
            "api_response": None,
            "cached": False
        }
        
        # Check cache first
        cache_key = location_description.lower().strip()
        if cache_key in self.geocoding_cache:
            cached_result = self.geocoding_cache[cache_key]
            result.update(cached_result)
            result["cached"] = True
            print(f"   ✓ Using cached result for '{location_description}'")
            return result
        
        # Respect rate limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        print(f"   → Google Geocoding: '{location_description}'...")
        
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': location_description,
            'key': self.google_api_key
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            self.last_request_time = time.time()
            
            response.raise_for_status()
            data = response.json()
            result["api_response"] = data
            
            if data['status'] == 'OK' and data['results']:
                location_data = data['results'][0]
                coords = location_data['geometry']['location']
                lat, lon = coords['lat'], coords['lng']
                
                # Validate coordinates
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    result.update({
                        "success": True,
                        "coordinates": {"latitude": lat, "longitude": lon},
                        "formatted_address": location_data.get('formatted_address')
                    })
                    print(f"   ✓ Found coordinates: ({lat}, {lon})")
                    
                    # Cache successful result
                    cache_data = {
                        "success": True,
                        "coordinates": {"latitude": lat, "longitude": lon},
                        "formatted_address": location_data.get('formatted_address')
                    }
                    self.geocoding_cache[cache_key] = cache_data
                else:
                    result["error"] = f"Invalid coordinate ranges: lat={lat}, lon={lon}"
                    print(f"   ✗ Invalid coordinate ranges")
            else:
                result["error"] = f"API Status: {data['status']}"
                if 'error_message' in data:
                    result["error"] += f" - {data['error_message']}"
                print(f"   ✗ API Error: {result['error']}")
                
        except requests.exceptions.RequestException as e:
            result["error"] = f"Request error: {str(e)}"
            print(f"   ✗ Request failed: {e}")
        except (KeyError, IndexError) as e:
            result["error"] = f"Response parsing error: {str(e)}"
            print(f"   ✗ Could not parse response: {e}")
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            print(f"   ✗ Unexpected error: {e}")
        
        return result
    
    def parse_final_location_response(self, response: str) -> Dict[str, Any]:
        """
        Parse final location response supporting both JSON and text formats.
        Returns comprehensive result with all attempts and metadata.
        NO GEOCODING - Only direct coordinate parsing for offline environments.
        
        Returns:
            Dict containing final coordinates, method used, and errors
        """
        parsing_result = {
            "ai_response": response,
            "parsing_attempts": [],
            "final_coordinates": None,
            "method_used": None,
            "success": False
        }
        
        try:
            # Extract JSON from markdown code blocks if present
            json_text = response.strip()
            if json_text.startswith('```json') and json_text.endswith('```'):
                json_text = json_text[7:-3].strip()  # Remove ```json and ```
            elif json_text.startswith('```') and json_text.endswith('```'):
                json_text = json_text[3:-3].strip()  # Remove ``` and ```
            
            # Try to parse as JSON
            data = json.loads(json_text)
            
            # Handle unified format with final_guess
            final_guess = data.get('final_guess', {})
            method = final_guess.get('method', '').lower()
            
            # Handle coordinates method (direct coordinates only)
            if method == 'coordinates' and final_guess.get('coordinates'):
                coords_data = final_guess['coordinates']
                if isinstance(coords_data, dict) and 'latitude' in coords_data and 'longitude' in coords_data:
                    lat, lon = coords_data['latitude'], coords_data['longitude']
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        parsing_result.update({
                            "final_coordinates": {"latitude": lat, "longitude": lon},
                            "method_used": "unified_json_direct_coordinates",
                            "success": True
                        })
                        parsing_result["parsing_attempts"].append({
                            "method": "unified_json_direct_coordinates", 
                            "success": True,
                            "result": {"latitude": lat, "longitude": lon}
                        })
                        print(f"   ✓ Unified JSON direct coordinates: ({lat}, {lon})")
                        return parsing_result
            
            # Handle detailed_description method OR coordinates method with location_description
            if (method == 'detailed_description' and final_guess.get('location_description')) or \
               (method == 'coordinates' and final_guess.get('location_description') and not final_guess.get('coordinates')):
                location_desc = final_guess['location_description'].strip()
                print(f"   → Unified JSON location description: '{location_desc}'")
                print(f"   ! Saving description for later geocoding processing")
                
                parsing_result["parsing_attempts"].append({
                    "method": "unified_json_detailed_description",
                    "success": True,
                    "description": location_desc,
                    "note": "Description saved for batch geocoding"
                })
                
                # Save the description as final result for batch processing
                parsing_result.update({
                    "final_coordinates": None,
                    "location_description": location_desc,
                    "method_used": "unified_json_location_description", 
                    "success": True,
                    "needs_geocoding": True
                })
                return parsing_result
                        
        except (json.JSONDecodeError, KeyError, TypeError):
            # Not valid JSON, continue with text parsing
            pass
        
        # Fallback to text-based parsing for backwards compatibility
        parsing_result["parsing_attempts"].append({
            "method": "json_parsing",
            "success": False,
            "error": "Not valid JSON or missing required fields"
        })
        
        # Try direct coordinates from text
        direct_coords = self.parse_coordinates_from_response(response)
        if direct_coords:
            lat, lon = direct_coords
            parsing_result.update({
                "final_coordinates": {"latitude": lat, "longitude": lon},
                "method_used": "text_direct_coordinates",
                "success": True
            })
            parsing_result["parsing_attempts"].append({
                "method": "text_direct_coordinates",
                "success": True,
                "result": {"latitude": lat, "longitude": lon}
            })
            print(f"   ✓ Text direct coordinates: {direct_coords}")
            return parsing_result
        
        print(f"   ✗ Could not parse direct coordinates from response (geocoding disabled)")
        return parsing_result
    
    def geocode_location(self, location_description: str) -> Optional[Tuple[float, float]]:
        """
        Converts a location description to coordinates using Nominatim geocoding.
        
        Args:
            location_description: Text description of location (e.g., "Times Square, New York")
            
        Returns:
            Tuple of (latitude, longitude) or None if geocoding fails
        """
        print(f"Geocoding location: '{location_description}'...")
        geolocator = Nominatim(user_agent="geo_aot_geoguess")
        
        try:
            location = geolocator.geocode(location_description)
            if location:
                lat, lon = location.latitude, location.longitude
                print(f"   ✓ Found coordinates: ({lat}, {lon})")
                # Basic validation for coordinate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
                else:
                    print(f"   ✗ Invalid coordinate ranges")
            else:
                print(f"   ✗ No location found")
        except Exception as e:
            print(f"   ✗ Geocoding failed: {e}")
        
        return None
    
    def parse_location_description_response(self, response: str) -> Optional[Tuple[str, Tuple[float, float]]]:
        """
        Parse AI response for location description and attempt geocoding.
        
        Returns:
            Tuple of (description, coordinates) or None if parsing/geocoding fails
        """
        # Look for "Location: [description]" format
        location_patterns = [
            r"Location:\s*(.+?)(?:\n|$)",
            r"I believe this is\s*([^\.]+?)(?:\s+based|\s+due|\.|$)",
            r"This appears to be\s*([^\.]+?)(?:\s+based|\s+due|\.|$)", 
            r"This looks like\s*([^\.]+?)(?:\s+based|\s+due|\.|$)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                location_desc = match.group(1).strip()
                coordinates = self.geocode_location(location_desc)
                if coordinates:
                    return (location_desc, coordinates)
        
        return None
    
    def reset_conversation(self):
        """Reset conversation history and state."""
        self.conversation_history = []
        self.step_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.token_call_records = []
        self._call_counter = 0

    def get_token_usage(self) -> Dict[str, Any]:
        """Get total token usage for this session with per-call breakdown."""
        return {
            # Backward compatible
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            # New per-call data
            "call_breakdown": self.token_call_records,
            "total_calls": len(self.token_call_records),
            "summary": {
                "navigation_calls": len([r for r in self.token_call_records if r["call_type"] == "navigation"]),
                "navigation_input_tokens": sum(r["input_tokens"] for r in self.token_call_records if r["call_type"] == "navigation"),
                "navigation_output_tokens": sum(r["output_tokens"] for r in self.token_call_records if r["call_type"] == "navigation"),
                "final_guess_calls": len([r for r in self.token_call_records if r["call_type"] == "final_guess"]),
                "final_guess_input_tokens": sum(r["input_tokens"] for r in self.token_call_records if r["call_type"] == "final_guess"),
                "final_guess_output_tokens": sum(r["output_tokens"] for r in self.token_call_records if r["call_type"] == "final_guess"),
            }
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state."""
        return {
            "step_count": self.step_count,
            "messages_count": len(self.conversation_history),
            "last_message": self.conversation_history[-1] if self.conversation_history else None,
            "model_config": {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        }
    
    def export_conversation_log(self, filepath: str):
        """Export conversation history to JSON file."""
        
        export_data = {
            "model_config": {
                "model": self.model,
                "temperature": self.temperature, 
                "max_tokens": self.max_tokens,
                "base_url": self.base_url
            },
            "conversation_history": self.conversation_history,
            "summary": self.get_conversation_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
