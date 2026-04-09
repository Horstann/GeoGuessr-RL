import json
import os
import base64
import re
import math
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from aot_ai_conversation import GeoAoTAI

class GeoAoTGuesser:
    def __init__(self, json_file, pano_folder, ai_config=None, ai_keys=None, debug=False, max_steps=10):
        self.json_file = json_file
        self.pano_folder = pano_folder
        self.graph_data = None
        self.nodes = {}
        self.current_node_id = None
        self.original_node_id = None
        self.current_rotation = 0  # Current manual rotation in degrees
        self.action_history = []
        self.debug = debug
        self.action_counter = 0
        self.max_steps = max_steps
        self.step_count = 0
        
        # Create debug folder if needed
        if self.debug:
            self.debug_folder = "actions"
            os.makedirs(self.debug_folder, exist_ok=True)
            print(f"Debug mode enabled. Saving to: {self.debug_folder}/")
            
            # Initialize debug log
            self.debug_log = {
                "session_start": self.__get_timestamp(),
                "json_file": json_file,
                "pano_folder": pano_folder,
                "actions": []
            }
        
        # AI conversation system setup
        if ai_config and ai_keys:
            try:
                self.ai_conversation = GeoAoTAI(
                    ai_config=ai_config,
                    ai_keys=ai_keys
                )
            except Exception as e:
                self.ai_conversation = None
                print(f"Warning: Failed to initialize AI conversation system: {e}")
                print("AI interaction disabled.")
        else:
            self.ai_conversation = None
            print("Warning: No AI configuration provided. AI interaction disabled.")
        
        # Colors for different arrows
        self.arrow_colors = {
            0: ("red", "#FF0000"),
            1: ("blue", "#0000FF"), 
            2: ("green", "#00FF00"),
            3: ("yellow", "#FFFF00"),
            4: ("purple", "#FF00FF"),
            5: ("orange", "#FFA500"),
            6: ("cyan", "#00FFFF"),
            7: ("white", "#FFFFFF")
        }
        
        self.load_graph_data()
    
    def __get_timestamp(self):
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def save_debug_info(self, composite_image, ai_response, chosen_action, available_moves):
        """Save debug information including panorama and action data"""
        if not self.debug:
            return
        
        self.action_counter += 1
        action_id = f"action_{self.action_counter:03d}"
        
        # Create action-specific folder
        action_folder = os.path.join(self.debug_folder, action_id)
        os.makedirs(action_folder, exist_ok=True)
        
        # Save panorama with action overlay
        pano_with_action = self.add_action_overlay(composite_image, chosen_action, ai_response)
        pano_path = os.path.join(action_folder, f"{action_id}_panorama.png")
        pano_with_action.save(pano_path)
        
        # Prepare action data
        action_data = {
            "action_id": action_id,
            "timestamp": self.__get_timestamp(),
            "current_node": self.current_node_id,
            "current_heading": self.current_rotation,
            "location_info": self.get_current_location_info(),
            "available_moves": [
                {
                    "color": self.arrow_colors.get(i, ("unknown", ""))[0],
                    "target_id": move['target_id'],
                    "direction_rad": move['direction']
                }
                for i, move in enumerate(available_moves)
            ],
            "ai_response": ai_response,
            "chosen_action": chosen_action,
            "action_history": self.action_history.copy()
        }
        
        # Save action data as JSON
        action_json_path = os.path.join(action_folder, f"{action_id}_data.json")
        with open(action_json_path, 'w') as f:
            json.dump(action_data, f, indent=2)
        
        # Add to debug log
        self.debug_log["actions"].append(action_data)
        
        # Save updated session log
        session_log_path = os.path.join(self.debug_folder, "session_log.json")
        with open(session_log_path, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
        
        print(f"Debug info saved: {action_folder}/")
    
    def add_action_overlay(self, composite_image, chosen_action, ai_response):
        """Add AI's chosen action as text overlay at top right"""
        img = composite_image.copy()
        draw = ImageDraw.Draw(img)
        
        # Prepare action text
        action_text = f"AI Action: {chosen_action.get('type', 'unknown').upper()}"
        if chosen_action['type'] == 'rotate':
            action_text += f" {chosen_action.get('degrees', 0)}°"
        elif chosen_action['type'] == 'move':
            action_text += f" to {chosen_action.get('color', 'unknown').upper()}"
        
        # Add response summary (first 50 chars)
        response_summary = ai_response[:50] + "..." if len(ai_response) > 50 else ai_response
        
        # Text positioning (top right)
        img_width, img_height = img.size
        text_x = img_width - 20
        text_y = 20
        
        # Create background rectangle for text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox1 = draw.textbbox((0, 0), action_text, font=font)
        bbox2 = draw.textbbox((0, 0), response_summary, font=font)
        text_width = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0]) + 20
        text_height = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + 30
        
        # Draw background rectangle
        rect_x1 = text_x - text_width
        rect_y1 = text_y
        rect_x2 = text_x
        rect_y2 = text_y + text_height
        
        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], 
                      fill=(0, 0, 0, 180), outline=(255, 255, 255, 255), width=2)
        
        # Draw text
        draw.text((text_x - 10, text_y + 10), action_text, 
                 fill='white', anchor='ra', font=font)
        draw.text((text_x - 10, text_y + 30), f"Response: {response_summary}", 
                 fill='lightgray', anchor='ra', font=font)
        
        return img
    
    def load_graph_data(self):
        """Load graph data from JSON file"""
        with open(self.json_file, 'r') as f:
            self.graph_data = json.load(f)
        
        # Process nodes
        for node in self.graph_data['nodes']:
            self.nodes[node['pano_id']] = node
        
        # Set starting position
        self.current_node_id = self.graph_data.get('center_pano_id')
        self.original_node_id = self.current_node_id
        
    def get_ground_truth_from_json(self):
        """Extract ground truth coordinates from the center pano in JSON file"""
        if not self.graph_data or not self.current_node_id:
            return None
        
        center_node = self.nodes.get(self.current_node_id)
        if not center_node or 'coordinate' not in center_node:
            return None
        
        coords = center_node['coordinate']
        if 'lat' in coords and 'lon' in coords:
            return {
                'latitude': coords['lat'],
                'longitude': coords['lon']
            }
        
        return None
    
    def get_current_location_info(self):
        """Get information about current location relative to original"""
        if not self.current_node_id or not self.original_node_id:
            return "Unknown location"
        
        current = self.nodes[self.current_node_id]['coordinate']
        original = self.nodes[self.original_node_id]['coordinate']
        
        # Calculate relative distance
        from geopy.distance import geodesic
        
        current_point = (current['lat'], current['lon'])
        original_point = (original['lat'], original['lon'])
        
        distance = geodesic(original_point, current_point).meters
        
        # Calculate bearing manually
        lat1, lon1 = math.radians(original['lat']), math.radians(original['lon'])
        lat2, lon2 = math.radians(current['lat']), math.radians(current['lon'])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing_deg = math.degrees(math.atan2(y, x))
        bearing_deg = (bearing_deg + 360) % 360  # Normalize to 0-360
        
        return f"Distance from start: {distance:.1f}m, Bearing: {bearing_deg:.1f}°"
    
    def get_current_pano_heading(self):
        """Get the natural heading of the current panorama from JSON data"""
        if not self.current_node_id or self.current_node_id not in self.nodes:
            return 0
        
        node = self.nodes[self.current_node_id]
        if 'coordinate' in node and 'heading' in node['coordinate']:
            heading_radians = node['coordinate']['heading']
            return math.degrees(heading_radians)
        else:
            return 0
    
    def roll_panorama(self, image, yaw_degrees):
        """Roll panorama horizontally by yaw_degrees"""
        img_np = np.array(image)
        width = img_np.shape[1]
        shift = int(width * yaw_degrees / 360)
        
        if shift != 0:
            rolled_img_np = np.roll(img_np, shift=shift, axis=1)
            return Image.fromarray(rolled_img_np)
        return image
    
    def create_orientation_header(self, width=800, height=60):
        """Create header showing pano's natural orientation"""
        img = Image.new('RGB', (width, height), (100, 149, 237))
        draw = ImageDraw.Draw(img)
        
        # Dark background bar
        draw.rounded_rectangle([10, 10, width-10, height-10], radius=20, fill=(60, 60, 60))
        
        # Load fonts
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        center_x = width // 2
        
        # Cardinal directions
        directions = {
            0: 'N', 90: 'E', 180: 'S', 270: 'W',
            45: 'NE', 135: 'SE', 225: 'SW', 315: 'NW'
        }
        
        # Total heading includes natural pano heading + manual rotation
        total_heading = self.get_current_pano_heading() + self.current_rotation
        
        for angle_deg, label in directions.items():
            # Calculate position relative to total heading
            relative_angle = angle_deg - total_heading
            
            # Normalize to -180 to +180
            normalized_angle = ((relative_angle + 180) % 360) - 180
            x_pos = center_x + (normalized_angle * width * 0.8) / 360
            
            # Only draw if visible
            if 40 < x_pos < width - 40:
                if angle_deg % 90 == 0:  # Major directions
                    draw.line([(x_pos, 22), (x_pos, 32)], fill='white', width=3)
                    draw.text((x_pos, 42), label, fill='white', anchor='mm', font=font_large)
                else:  # Secondary directions
                    draw.line([(x_pos, 24), (x_pos, 30)], fill='lightgray', width=2)
                    draw.text((x_pos, 42), label, fill='lightgray', anchor='mm', font=font_small)
        
        # Center indicator pointing to where pano is facing
        triangle_points = [(center_x, 15), (center_x-8, 25), (center_x+8, 25)]
        draw.polygon(triangle_points, fill='red')
        
        return img
    
    def get_available_moves(self):
        """Get available moves from current position"""
        if not self.current_node_id:
            return []
        
        current_matrix_id = self.nodes[self.current_node_id]['matrix_id']
        adjacency_matrix = self.graph_data['adjacency_matrix']
        
        available_moves = []
        node_list = list(self.nodes.keys())
        
        for i, target_node_id in enumerate(node_list):
            if (current_matrix_id < len(adjacency_matrix) and 
                i < len(adjacency_matrix[current_matrix_id]) and
                adjacency_matrix[current_matrix_id][i] != -1):
                
                direction = adjacency_matrix[current_matrix_id][i]
                available_moves.append({
                    'target_id': target_node_id,
                    'direction': direction,
                    'color_index': len(available_moves)
                })
        
        return available_moves
    
    def draw_google_maps_arrow(self, draw, center_x, center_y, direction_deg, color_hex, size=40):
        """Draw Google Maps style directional arrow"""
        angle_rad = math.radians(direction_deg)
        
        # Calculate arrow tip position
        tip_x = center_x + size * math.sin(angle_rad)
        tip_y = center_y - size * math.cos(angle_rad)  # Negative because Y increases downward
        
        # Calculate arrow body points
        body_length = size * 0.7
        body_x = center_x + body_length * math.sin(angle_rad)
        body_y = center_y - body_length * math.cos(angle_rad)
        
        # Calculate arrow head points
        head_size = size * 0.3
        perpendicular_angle = angle_rad + math.pi/2
        
        left_x = body_x + head_size * math.sin(perpendicular_angle)
        left_y = body_y - head_size * math.cos(perpendicular_angle)
        right_x = body_x - head_size * math.sin(perpendicular_angle)
        right_y = body_y + head_size * math.cos(perpendicular_angle)
        
        # Draw arrow shaft (thick line)
        draw.line([(center_x, center_y), (body_x, body_y)], fill=color_hex, width=8)
        
        # Draw arrow head (triangle)
        arrow_head = [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)]
        draw.polygon(arrow_head, fill=color_hex, outline='black', width=2)
        
    def draw_arrows_on_panorama(self, pano_image, available_moves):
        """Draw Google Maps style arrows on ellipse near center"""
        img = pano_image.copy()
        draw = ImageDraw.Draw(img)
        
        width, height = img.size
        
        # Define ellipse for arrow placement (near center)
        ellipse_center_x = width // 2
        ellipse_center_y = height // 2
        ellipse_radius_x = width // 8  # Horizontal radius
        ellipse_radius_y = height // 10  # Vertical radius
        
        for i, move in enumerate(available_moves):
            direction_rad = move['direction']  # Direction in radians from JSON
            color_name, color_hex = self.arrow_colors.get(i, ("white", "#FFFFFF"))
            
            # Convert direction to degrees
            direction_deg = math.degrees(direction_rad)
            
            # Adjust direction relative to pano's natural heading + manual rotation
            total_heading = self.get_current_pano_heading() + self.current_rotation
            adjusted_direction = direction_deg - total_heading
            adjusted_direction = adjusted_direction % 360
            
            # Calculate position on ellipse
            angle_for_ellipse = math.radians(adjusted_direction)
            arrow_center_x = ellipse_center_x + ellipse_radius_x * math.sin(angle_for_ellipse)
            arrow_center_y = ellipse_center_y - ellipse_radius_y * math.cos(angle_for_ellipse)
            
            # Draw Google Maps style arrow
            self.draw_google_maps_arrow(draw, arrow_center_x, arrow_center_y, adjusted_direction, color_hex)
        
        return img
    
    def create_composite_view(self):
        """Create the complete view with panorama, arrows, and header"""
        # Load current panorama
        pano_path = os.path.join(self.pano_folder, f"{self.current_node_id}.png")
        if not os.path.exists(pano_path):
            print(f"Panorama not found: {pano_path}")
            return None
        
        pano_image = Image.open(pano_path)
        
        # Apply current rotation
        if self.current_rotation != 0:
            pano_image = self.roll_panorama(pano_image, self.current_rotation)
        
        # Get available moves and draw arrows
        available_moves = self.get_available_moves()
        pano_with_arrows = self.draw_arrows_on_panorama(pano_image, available_moves)
        
        # Create orientation header
        header = self.create_orientation_header(pano_with_arrows.width, 60)
        
        # Combine header and panorama
        total_height = header.height + pano_with_arrows.height
        composite = Image.new('RGB', (pano_with_arrows.width, total_height))
        composite.paste(header, (0, 0))
        composite.paste(pano_with_arrows, (0, header.height))
        
        return composite, available_moves
    
    def parse_action(self, ai_response, available_moves):
        """Parse AI response for actions using AI conversation system"""
        if not self.ai_conversation:
            # Fallback to basic parsing if AI conversation not available
            return self._basic_parse_action(ai_response, available_moves)
        
        parsed_action = self.ai_conversation.parse_action_response(ai_response)
        
        # If it's a move action, find the matching move from available_moves
        if parsed_action['type'] == 'move':
            color = parsed_action['color']
            for i, move in enumerate(available_moves):
                move_color = self.arrow_colors.get(i, ("", ""))[0]
                if color == move_color:
                    return {'type': 'move', 'target_id': move['target_id'], 'color': color}
        
        return parsed_action
    
    def _basic_parse_action(self, ai_response, available_moves):
        """Basic action parsing fallback"""
        response = ai_response.lower().strip()
        
        # Parse rotation command (handle both positive and negative numbers)
        rotate_match = re.search(r'rotate\s+([-+]?\d+)\s*degrees?', response)
        if rotate_match:
            degrees = int(rotate_match.group(1))
            return {'type': 'rotate', 'degrees': degrees}
        
        # Parse move command by color
        color_patterns = [
            r'move\s+to\s+(\w+)',
            r'go\s+to\s+(\w+)',
            r'take\s+(\w+)',
            r'(\w+)\s+arrow'
        ]
        
        for pattern in color_patterns:
            move_match = re.search(pattern, response)
            if move_match:
                color = move_match.group(1).lower()
                # Find matching move
                for i, move in enumerate(available_moves):
                    move_color = self.arrow_colors.get(i, ("", ""))[0]
                    if color == move_color:
                        return {'type': 'move', 'target_id': move['target_id'], 'color': color}
        
        # Parse back to original command
        if 'back' in response or 'original' in response or 'start' in response:
            return {'type': 'back'}
        
        return {'type': 'unknown', 'response': response}
    
    def execute_action(self, action):
        """Execute the parsed action"""
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            return f"Maximum steps ({self.max_steps}) reached. No more actions allowed."
        
        if action['type'] == 'rotate':
            degrees = action['degrees']
            self.current_rotation = (self.current_rotation + degrees) % 360
            self.step_count += 1
            self.action_history.append(f"Rotated {degrees} degrees")
            return f"Rotated view by {degrees} degrees. Current heading: {self.current_rotation}°"
        
        elif action['type'] == 'move':
            target_id = action['target_id']
            if target_id in self.nodes:
                self.current_node_id = target_id
                self.current_rotation = 0  # Reset rotation at new location
                self.step_count += 1
                self.action_history.append(f"Moved to {action['color']} arrow")
                return f"Moved to {action['color']} arrow location. {self.get_current_location_info()}"
            else:
                return "Error: Invalid move target"
        
        elif action['type'] == 'back':
            self.current_node_id = self.original_node_id
            self.current_rotation = 0
            self.step_count += 1
            self.action_history.append("Returned to original location")
            return "Returned to original starting location"
        
        elif action['type'] == 'guess':
            # AI has made an early guess - trigger final location processing
            self.step_count += 1  # Count guessing as an action step
            self.action_history.append(f"Made early location guess (step {self.step_count})")
            return "EARLY_GUESS_MADE"  # Special return code for early termination
        
        else:
            return f"Unknown command. Available actions: rotate X degrees, move to [color], back to original, guess location"
    
    def is_max_steps_reached(self):
        """Check if maximum steps have been reached"""
        return self.step_count >= self.max_steps
    
    def get_ai_response(self, composite_image, available_moves, current_step):
        """Get AI response for the current view using the conversation system"""
        if not self.ai_conversation:
            return "AI conversation system not configured"
        
        # Prepare move data for AI
        move_data = []
        for i, move in enumerate(available_moves):
            color_name = self.arrow_colors.get(i, ("unknown", ""))[0]
            move_data.append({
                "color": color_name,
                "target_node": move.get("target_id", "unknown"),
                "direction": move.get("direction", 0)
            })
        
        return self.ai_conversation.get_navigation_decision(
            composite_image=composite_image,
            current_step=current_step,
            current_node=self.current_node_id,
            current_heading=self.get_current_pano_heading(),
            current_rotation=self.current_rotation,
            available_moves=move_data,
            location_info=self.get_current_location_info(),
            max_steps=self.max_steps
        )
    
    def get_final_location_guess(self, composite_image):
        """Get final location coordinates guess from AI after exploration"""
        if not self.ai_conversation:
            return None, "AI conversation system not configured"
        
        ai_response = self.ai_conversation.get_final_location_guess(
            composite_image=composite_image,
            exploration_summary=self.action_history,
            current_node=self.current_node_id,
            current_heading=self.get_current_pano_heading(),
            current_rotation=self.current_rotation,
            location_info=self.get_current_location_info()
        )
        
        return ai_response, None
    
    def parse_coordinates_from_text(self, text):
        """Parse coordinates from AI response text using AI conversation system"""
        if not self.ai_conversation:
            return None
        
        return self.ai_conversation.parse_final_location_response(text)
    
    def create_comprehensive_output(self, ground_truth_coords=None, step_results=None):
        """Create comprehensive JSON output including GeoAoT and final result"""
        # Auto-extract ground truth from JSON if not provided
        if ground_truth_coords is None:
            ground_truth_coords = self.get_ground_truth_from_json()
        
        # Get final composite view
        composite_image, available_moves = self.create_composite_view()
        
        # Check for early guess in step results
        early_guess_response = None
        early_guess_step = None
        if step_results:
            for step in step_results:
                if step.get("early_guess") and step.get("ai_response"):
                    early_guess_response = step["ai_response"]
                    early_guess_step = step["step"]
                    break
        
        # Get final location guess (use early guess if available)
        if early_guess_response:
            print(f"Using early guess from step {early_guess_step}")
            ai_response, error = early_guess_response, None
        else:
            ai_response, error = self.get_final_location_guess(composite_image)
        
        # Parse coordinates with comprehensive results
        parsing_result = None
        pred_coords = None
        location_description = None
        if ai_response and not error:
            parsing_result = self.parse_coordinates_from_text(ai_response)
            if parsing_result and parsing_result.get("success"):
                pred_coords = parsing_result.get("final_coordinates")
                location_description = parsing_result.get("location_description")
        
        # Calculate distance if ground truth provided
        distance_km = None
        if ground_truth_coords and pred_coords:
            from geopy.distance import geodesic
            gt_point = (ground_truth_coords["latitude"], ground_truth_coords["longitude"])
            pred_point = (pred_coords["latitude"], pred_coords["longitude"])
            distance_km = round(geodesic(gt_point, pred_point).kilometers, 2)

        # Get token usage from AI conversation
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if self.ai_conversation:
            token_usage = self.ai_conversation.get_token_usage()

        # Create comprehensive output
        output = {
            "geo_aot": {
                "max_steps": self.max_steps,
                "steps_taken": self.step_count,
                "action_history": self.action_history,
                "starting_node": self.original_node_id,
                "final_node": self.current_node_id,
                "final_rotation": self.current_rotation,
                "exploration_complete": self.step_count >= self.max_steps - 1 or len(available_moves) == 0  
            },
            "final_result": {
                "ai_response": ai_response,
                "pred_coords": pred_coords,
                "location_description": location_description,
                "needs_geocoding": parsing_result.get("needs_geocoding", False) if parsing_result else False,
                "error": error,
                "parsing_details": parsing_result,
                "early_guess": {
                    "was_early_guess": early_guess_response is not None,
                    "guess_step": early_guess_step if early_guess_response else None,
                    "steps_saved": (self.max_steps - early_guess_step) if early_guess_step else 0
                },
                "geocoding_metadata": {
                    "total_attempts": len(parsing_result.get("parsing_attempts", [])) if parsing_result else 0,
                    "geocoding_calls": len(parsing_result.get("geocoding_results", [])) if parsing_result else 0,
                    "method_used": parsing_result.get("method_used") if parsing_result else None
                }
            },
            "token_usage": token_usage
        }
        
        # Add ground truth and distance if provided
        if ground_truth_coords:
            output["ground_truth"] = {
                "gt_coords": ground_truth_coords,
                "distance_km": distance_km
            }
        
        return output
    
    def run_geo_aot_session(self, ground_truth_coords=None, interactive=False):
        """Run a complete GeoAoT (Action of Thought) session until max steps or completion"""
        # Auto-extract ground truth from JSON if not provided
        if ground_truth_coords is None:
            ground_truth_coords = self.get_ground_truth_from_json()
            if ground_truth_coords:
                print(f"Ground truth auto-extracted from JSON: ({ground_truth_coords['latitude']:.5f}, {ground_truth_coords['longitude']:.5f})")
            else:
                print("No ground truth coordinates found in JSON file")
        
        conversation_history = ""
        step_results = []
        
        while self.step_count < self.max_steps - 1:
            # Create current view
            composite_image, available_moves = self.create_composite_view()
            if not composite_image:
                print("Error creating composite view")
                break
                
            # Record current state before action
            current_step_info = {
                "step": self.step_count + 1,
                "current_node": self.current_node_id,
                "current_heading": self.get_current_pano_heading(),
                "current_rotation": self.current_rotation,
                "available_moves": [
                    {
                        "color": self.arrow_colors.get(i, ("unknown", ""))[0],
                        "target_node": move["target_id"],
                        "direction": move["direction"]
                    } for i, move in enumerate(available_moves)
                ],
                "finished": False,
                "next_action": None,
                "ai_response": None,
                "execution_result": None
            }
            
            # Save debug image if enabled
            if self.debug:
                debug_path = os.path.join(self.debug_folder, f"step_{self.step_count + 1:03d}_view.png")
                composite_image.save(debug_path)
                print(f"Debug image saved: {debug_path}")
            
            if interactive:
                # Interactive mode - wait for user input
                print("\\nAvailable actions:")
                print("1. ROTATE X DEGREES")
                print("2. MOVE TO COLOR:", [move["color"] for move in current_step_info["available_moves"]])
                print("3. BACK TO ORIGINAL")
                print("4. FINISH - Get final location guess")
                
                user_input = input("\\nWhat would you like to do? ")
                current_step_info["ai_response"] = user_input
                
                if user_input.lower() in ['finish', 'done', 'guess']:
                    print("Finishing exploration and getting final location guess...")
                    current_step_info["finished"] = True
                    current_step_info["next_action"] = None
                    step_results.append(current_step_info)
                    break
                
                # Parse user action
                action = self.parse_action(user_input, available_moves)
                print(f"Parsed action: {action}")
                
            else:
                # AI mode - get AI response
                ai_response = self.get_ai_response(composite_image, available_moves, self.step_count + 1)
                current_step_info["ai_response"] = ai_response
                
                # Parse AI action
                action = self.parse_action(ai_response, available_moves)
                conversation_history += f"\\nStep {self.step_count + 1}: {ai_response}"
            
            # Record the planned action
            current_step_info["next_action"] = action
            
            # Execute action
            if action['type'] != 'unknown':
                result = self.execute_action(action)
                current_step_info["execution_result"] = result
                
                # Check for early guess termination
                if result == "EARLY_GUESS_MADE":
                    current_step_info["finished"] = True
                    current_step_info["early_guess"] = True
                    step_results.append(current_step_info)
                    break
                
            else:
                print("Unknown action, skipping...")
                current_step_info["execution_result"] = "Unknown action, skipped"
                if not interactive:
                    # In AI mode, if we can't parse the action, break to avoid infinite loop
                    print("AI provided unparseable action, finishing session...")
                    current_step_info["finished"] = True
                    step_results.append(current_step_info)
                    break
            
            # Add the completed step to results
            step_results.append(current_step_info)
            
            # Check if no moves available (exploration complete)
            if len(available_moves) == 0:
                print("No more moves available, exploration complete")
                break
                
        print(f"\\nExploration complete! Took {self.step_count} steps out of {self.max_steps} maximum")
        
        # Add final step showing completion state
        final_composite_image, final_available_moves = self.create_composite_view()
        if final_composite_image:
            final_step_info = {
                "step": self.step_count + 1,
                "current_node": self.current_node_id,
                "current_heading": self.get_current_pano_heading(),
                "current_rotation": self.current_rotation,
                "available_moves": [
                    {
                        "color": self.arrow_colors.get(i, ("unknown", ""))[0],
                        "target_node": move["target_id"],
                        "direction": move["direction"]
                    } for i, move in enumerate(final_available_moves)
                ],
                "finished": True,
                "next_action": None,
                "ai_response": "Exploration complete - ready for final location guess",
                "execution_result": "Session completed"
            }
            step_results.append(final_step_info)
        
        # Create comprehensive output
        final_output = self.create_comprehensive_output(ground_truth_coords, step_results)
        
        # Add step-by-step results
        final_output["step_details"] = step_results
        
        return final_output


def main():
    parser = argparse.ArgumentParser(description='GeoAoT (Action of Thought) Geo-Guessing System')
    parser.add_argument('--json_file', required=True, help='Path to graph JSON file')
    parser.add_argument('--pano_folder', required=True, help='Path to panorama images folder')
    parser.add_argument('--test', action='store_true', help='Run single image test')
    parser.add_argument('--debug', action='store_true', help='Run debug test with AI interaction')
    parser.add_argument('--api_key', help='OpenAI API key')
    parser.add_argument('--base_url', help='OpenAI base URL')
    
    args = parser.parse_args()
    
    if args.debug:
        test_single_image_debug(args.json_file, args.pano_folder)
    elif args.test:
        test_single_image(args.json_file, args.pano_folder)
    else:
        print("Available modes:")
        print("  --test     : Basic functionality test")
        print("  --debug    : AI interaction test with debug saving")
        print("Use one of these flags to run the system.")

if __name__ == "__main__":
    main()
