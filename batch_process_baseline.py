import hydra
from omegaconf import DictConfig
from ai_client import get_openai_client
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import base64
from PIL import Image, ImageDraw, ImageFont
import math
from geopy.distance import geodesic

class BaselineGeoGuesser:
    """
    Baseline geolocation guesser that makes direct predictions without GeoAoT (Action of Thought) exploration.
    Uses the same JSON format as the GeoAoT system for consistent comparison.
    """
    
    def __init__(self, json_file_path: str, pano_folder: str, 
                 ai_config: DictConfig, ai_keys: DictConfig, debug: bool = False):
        self.json_file_path = json_file_path
        self.pano_folder = pano_folder
        self.debug = debug
        
        # Initialize AI client
        self.client = get_openai_client(ai_config=ai_config, ai_keys=ai_keys)
        self.model = ai_config.model
        self.max_tokens = ai_config.max_tokens
        self.temperature = ai_config.temperature

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Load graph data
        self.graph_data = None
        self.nodes = {}
        self.center_node_id = None
        self.load_graph_data()
    
    def load_graph_data(self):
        """Load graph data from JSON file"""
        with open(self.json_file_path, 'r') as f:
            self.graph_data = json.load(f)
        
        # Process nodes
        for node in self.graph_data['nodes']:
            self.nodes[node['pano_id']] = node
        
        # Set center node
        self.center_node_id = self.graph_data.get('center_pano_id')
        
        if self.debug:
            print(f"Loaded graph with {len(self.nodes)} nodes, center: {self.center_node_id}")
    
    def get_ground_truth_from_json(self):
        """Extract ground truth coordinates from the center pano in JSON file"""
        if not self.graph_data or not self.center_node_id:
            return None
        
        center_node = self.nodes.get(self.center_node_id)
        if not center_node or 'coordinate' not in center_node:
            return None
        
        coords = center_node['coordinate']
        if 'lat' in coords and 'lon' in coords:
            return {
                'latitude': coords['lat'],
                'longitude': coords['lon']
            }
        
        return None
    
    def get_current_pano_heading(self):
        """Get the natural heading of the center panorama from JSON data"""
        if not self.center_node_id or self.center_node_id not in self.nodes:
            return 0
        
        node = self.nodes[self.center_node_id]
        if 'coordinate' in node and 'heading' in node['coordinate']:
            heading_radians = node['coordinate']['heading']
            return math.degrees(heading_radians)
        else:
            return 0
    
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
        total_heading = self.get_current_pano_heading() + 0  # No manual rotation in baseline
        
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
    
    def create_baseline_view(self):
        """Create the baseline view with just panorama and header (no arrows)"""
        # Load center panorama
        pano_path = os.path.join(self.pano_folder, f"{self.center_node_id}.png")
        if not os.path.exists(pano_path):
            print(f"Panorama not found: {pano_path}")
            return None
        
        pano_image = Image.open(pano_path)
        
        # Create orientation header
        header = self.create_orientation_header(pano_image.width, 60)
        
        # Combine header and panorama
        total_height = header.height + pano_image.height
        composite = Image.new('RGB', (pano_image.width, total_height))
        composite.paste(header, (0, 0))
        composite.paste(pano_image, (0, header.height))
        
        return composite
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API"""
        if isinstance(image_path, str):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # Handle PIL Image objects
            import io
            img_buffer = io.BytesIO()
            image_path.save(img_buffer, format='PNG')
            return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    def get_direct_prediction(self, composite_image) -> str:
        """
        Get direct location prediction from AI using the same JSON format as GeoAoT system.
        This is the baseline approach - no exploration, just direct prediction from single image.
        """

        # Use the same system prompt structure as GeoAoT but simplified for direct prediction
        system_prompt = """IMPORTANT: Always respond in this unified JSON format:
{
  "observations": "What you see in the image in one or two short sentences",
  "confidence": "low" | "medium" | "high",
  "action": "guess",
  "final_guess": {
    "method": "detailed_description" | "coordinates",
    "location_description": "[Address] [Street Name], [City], [State], [Country]",
    "coordinates": {"latitude": 34.0522, "longitude": -118.2437}
  }
}

Since this is a direct prediction (no exploration), always set action = "guess" and provide your final_guess.

When making final_guess:
- method "detailed_description": MUST use EXACT format "[Address] [Street Name], [City], [State], [Country]" OR reduced specificity like "[City], [State], [Country]" OR "[State], [Country]" OR "[Country]". NO descriptive text, NO explanations, ONLY location names in this format. If absolutely no geographical clues available, use EXACTLY "FAIL TO PREDICT". Set coordinates to null.
- method "coordinates": Set coordinates to {"latitude": X, "longitude": Y}, set location_description to null.

CRITICAL: location_description must be ONLY location names in the specified format. NO descriptive phrases like "suburban area" or "appears to be". 

WARNING: Using "FAIL TO PREDICT" results in severe punishment. When uncertain, strongly consider using coordinates method instead."""
        
        user_text = "Analyze this Street View panorama and make your best guess about the location. Provide your response in the required JSON format."
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._encode_image(composite_image)}"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens or 0
                self.total_output_tokens += response.usage.completion_tokens or 0

            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI Error: {e}"

    def get_token_usage(self):
        """Get total token usage for this session."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }

    def parse_final_location_response(self, response: str) -> Dict[str, Any]:
        """
        Parse final location response using the same logic as GeoAoT system.
        Returns comprehensive result with all attempts and metadata.
        """
        parsing_result = {
            "ai_response": response,
            "parsing_attempts": [],
            "final_coordinates": None,
            "location_description": None,
            "method_used": None,
            "success": False,
            "needs_geocoding": False
        }
        
        try:
            # Extract JSON from markdown code blocks if present
            json_text = response.strip()
            if json_text.startswith('```json') and json_text.endswith('```'):
                json_text = json_text[7:-3].strip()
            elif json_text.startswith('```') and json_text.endswith('```'):
                json_text = json_text[3:-3].strip()
            
            # Try to parse as JSON
            data = json.loads(json_text)
            
            # Handle unified format with final_guess
            final_guess = data.get('final_guess', {})
            method = final_guess.get('method', '').lower()
            
            # Handle coordinates method (direct coordinates)
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
                        if self.debug:
                            print(f"   ✓ Direct coordinates: ({lat}, {lon})")
                        return parsing_result
            
            # Handle detailed_description method
            if method == 'detailed_description' and final_guess.get('location_description'):
                location_desc = final_guess['location_description'].strip()
                if self.debug:
                    print(f"   → Location description: '{location_desc}'")
                    print(f"   ! Saving description for geocoding")
                
                parsing_result["parsing_attempts"].append({
                    "method": "unified_json_detailed_description",
                    "success": True,
                    "description": location_desc,
                    "note": "Description saved for geocoding"
                })
                
                parsing_result.update({
                    "final_coordinates": None,
                    "location_description": location_desc,
                    "method_used": "unified_json_location_description", 
                    "success": True,
                    "needs_geocoding": True
                })
                return parsing_result
                        
        except (json.JSONDecodeError, KeyError, TypeError):
            # Not valid JSON, continue with fallback
            pass
        
        # Fallback parsing failed
        parsing_result["parsing_attempts"].append({
            "method": "json_parsing",
            "success": False,
            "error": "Not valid JSON or missing required fields"
        })
        
        if self.debug:
            print(f"   ✗ Could not parse response for direct coordinates")
        return parsing_result
    
    def run_baseline_prediction(self) -> Dict[str, Any]:
        """Run baseline direct prediction without exploration"""
        if self.debug:
            print(f"Running baseline direct prediction for {self.center_node_id}")
        
        # Get ground truth
        ground_truth_coords = self.get_ground_truth_from_json()
        
        # Create baseline view (just panorama with header, no arrows)
        composite_image = self.create_baseline_view()
        if not composite_image:
            return {"error": "Failed to create composite view"}
        
        # Get AI prediction
        ai_response = self.get_direct_prediction(composite_image)
        if self.debug:
            print(f"AI Response: {ai_response}")
        
        # Parse response
        parsing_result = self.parse_final_location_response(ai_response)
        pred_coords = parsing_result.get("final_coordinates")
        location_description = parsing_result.get("location_description")
        
        # Calculate distance if ground truth and prediction available
        distance_km = None
        if ground_truth_coords and pred_coords:
            gt_point = (ground_truth_coords["latitude"], ground_truth_coords["longitude"])
            pred_point = (pred_coords["latitude"], pred_coords["longitude"])
            distance_km = round(geodesic(gt_point, pred_point).kilometers, 2)
        
        # Create comprehensive output matching GeoAoT format
        output = {
            "geo_aot": {
                "max_steps": 1,  # Baseline uses only 1 step
                "steps_taken": 1,
                "action_history": ["Direct prediction without exploration"],
                "starting_node": self.center_node_id,
                "final_node": self.center_node_id,
                "final_rotation": 0,
                "exploration_complete": True
            },
            "final_result": {
                "ai_response": ai_response,
                "pred_coords": pred_coords,
                "location_description": location_description,
                "needs_geocoding": parsing_result.get("needs_geocoding", False),
                "error": None,
                "parsing_details": parsing_result,
                "early_guess": {
                    "was_early_guess": False,  # Baseline is always direct
                    "guess_step": None,
                    "steps_saved": 0
                },
                "geocoding_metadata": {
                    "total_attempts": len(parsing_result.get("parsing_attempts", [])),
                    "geocoding_calls": 0,  # No geocoding in batch processing
                    "method_used": parsing_result.get("method_used")
                }
            },
            "token_usage": self.get_token_usage()
        }
        
        # Add ground truth and distance if available
        if ground_truth_coords:
            output["ground_truth"] = {
                "gt_coords": ground_truth_coords,
                "distance_km": distance_km
            }
        
        return output


def process_single_file_baseline(json_file_path: str, pano_folder: str, output_dir: Path, 
                                ai_config: DictConfig, ai_keys: DictConfig, debug: bool) -> Dict[str, Any]:
    """Process a single JSON file with baseline direct prediction"""
    try:
        start_time = time.time()
        file_name = Path(json_file_path).stem
        print(f"Processing (Baseline): {file_name}")
        
        # Initialize baseline system
        guesser = BaselineGeoGuesser(
            json_file_path, 
            pano_folder,
            ai_config=ai_config,
            ai_keys=ai_keys,
            debug=debug
        )
        
        # Run baseline prediction
        result = guesser.run_baseline_prediction()
        
        # Save results to output folder with matching filename
        output_file = output_dir / f'{file_name}_baseline_results.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        processing_time = time.time() - start_time
        
        # Create summary for this file
        summary = {
            'file_name': file_name,
            'input_file': json_file_path,
            'output_file': str(output_file),
            'success': True,
            'processing_time': processing_time,
            'method': 'baseline_direct_prediction',
            'steps_taken': result['geo_aot']['steps_taken'],
            'max_steps': result['geo_aot']['max_steps'],
            'final_coordinates': result['final_result']['pred_coords'],
            'location_description': result['final_result'].get('location_description'),
            'needs_geocoding': result['final_result'].get('needs_geocoding', False),
            'distance_error': result.get('ground_truth', {}).get('distance_km'),
            'error': None
        }
        
        print(f"✓ Completed {file_name} in {processing_time:.2f}s (Baseline)")
        
        if result['final_result']['pred_coords']:
            coords = result['final_result']['pred_coords']
            print(f"  Final prediction: ({coords['latitude']}, {coords['longitude']})")
        
        if 'ground_truth' in result and result['ground_truth']['distance_km'] is not None:
            print(f"  Distance error: {result['ground_truth']['distance_km']} km")
            
        return summary
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed {file_name}: {error_msg}")
        
        return {
            'file_name': file_name,
            'input_file': json_file_path,
            'output_file': None,
            'success': False,
            'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
            'method': 'baseline_direct_prediction',
            'steps_taken': 0,
            'max_steps': 1,
            'final_coordinates': None,
            'location_description': None,
            'needs_geocoding': False,
            'distance_error': None,
            'error': error_msg
        }


def get_json_files(input_folder: str) -> List[str]:
    """Get all JSON files from input folder"""
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in: {input_folder}")
    
    return [str(f) for f in sorted(json_files)]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main baseline batch processing function"""
    print("=== Baseline Direct Prediction Batch Processing System ===")
    print(f"Config: {cfg}")
    
    input_folder = cfg.batch_process.input_graphs_folder
    pano_folder = cfg.batch_process.pano_folder
    debug = cfg.debug
    max_workers = cfg.batch_process.get('max_workers', 4)
    
    # Create output directory with baseline suffix
    output_dir = Path(cfg.output_folder + "_baseline")
    cfg.output_folder = os.path.join(output_dir, cfg.ai_config.model)
    output_dir = Path(cfg.output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Get all JSON files to process
    try:
        json_files = get_json_files(input_folder)
        print(f"Found {len(json_files)} JSON files to process with baseline method")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # Process files concurrently
    start_time = time.time()
    results = []
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Starting baseline batch processing with {max_workers} concurrent workers...")
        
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file_baseline, 
                json_file, pano_folder, output_dir,
                cfg.ai_config, cfg.ai_keys, debug
            ): json_file 
            for json_file in json_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                completed = len(results)
                total = len(json_files)
                print(f"Progress: {completed}/{total} files completed")
                
            except Exception as e:
                print(f"Exception processing {json_file}: {e}")
                results.append({
                    'file_name': Path(json_file).stem,
                    'input_file': json_file,
                    'output_file': None,
                    'success': False,
                    'processing_time': 0,
                    'method': 'baseline_direct_prediction',
                    'steps_taken': 0,
                    'max_steps': 1,
                    'final_coordinates': None,
                    'location_description': None,
                    'needs_geocoding': False,
                    'distance_error': None,
                    'error': str(e)
                })
    
    # Calculate final statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    successful_distances = [r['distance_error'] for r in results if r['success'] and r['distance_error'] is not None]
    avg_distance = sum(successful_distances) / len(successful_distances) if successful_distances else None
    
    # Geocoding statistics
    needs_geocoding = sum(1 for r in results if r['success'] and r.get('needs_geocoding', False))
    has_coordinates = sum(1 for r in results if r['success'] and r['final_coordinates'] is not None)
    has_descriptions = sum(1 for r in results if r['success'] and r.get('location_description') is not None)
    
    # Create batch summary
    batch_summary = {
        'batch_info': {
            'method': 'baseline_direct_prediction',
            'input_folder': input_folder,
            'output_folder': str(output_dir),
            'total_files': len(json_files),
            'max_workers': max_workers,
            'processing_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'statistics': {
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(json_files) * 100 if json_files else 0,
            'avg_processing_time_per_file': total_time / len(json_files) if json_files else 0,
            'avg_distance_error_km': avg_distance,
            'geocoding_stats': {
                'needs_geocoding': needs_geocoding,
                'has_coordinates': has_coordinates,
                'has_descriptions': has_descriptions,
                'direct_coords_rate': (has_coordinates / successful * 100) if successful > 0 else 0,
                'description_rate': (has_descriptions / successful * 100) if successful > 0 else 0
            }
        },
        'individual_results': results,
        'failed_files': [r for r in results if not r['success']]
    }
    
    # Save batch summary
    summary_file = output_dir / 'batch_summary_baseline.json'
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Print final summary
    print(f"\n=== Baseline Batch Processing Complete ===")
    print(f"Method: Direct prediction (no exploration)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Files processed: {len(json_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {batch_summary['statistics']['success_rate']:.1f}%")
    print(f"Average processing time per file: {batch_summary['statistics']['avg_processing_time_per_file']:.2f}s")
    
    if avg_distance is not None:
        print(f"Average distance error: {avg_distance:.2f} km")
    
    print(f"Results saved to: {output_dir}")
    print(f"Batch summary: {summary_file}")
    
    if failed > 0:
        print(f"\nFailed files:")
        for result in batch_summary['failed_files']:
            print(f"  - {result['file_name']}: {result['error']}")


if __name__ == "__main__":
    main()
