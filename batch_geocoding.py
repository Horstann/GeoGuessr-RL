#!/usr/bin/env python3
"""
Batch Geocoding System for GeoAoT (Action of Thought) Results
Paper: Learning to Wander: Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning
This script processes batch result files and geocodes location descriptions to coordinates.
"""

import json
import time
import argparse
from threading import Lock
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

class BatchGeocoder:
    """
    Post-processing geocoding system for batch GeoAoT (Action of Thought) results.
    Processes files that contain location descriptions and converts them to coordinates.
    """
    
    def __init__(self, google_api_key: Optional[str] = None, use_google: Optional[bool] = True):
        self.google_api_key = google_api_key
        self.geocoding_cache = {}
        self._cache_lock = Lock()
        self.cache_path = Path(__file__).resolve().parent / 'data' / 'geocoding_cache.jsonl'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            with self.cache_path.open('r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, start=1):
                    # Every append includes a newline; a missing one indicates
                    # an incomplete write, even if the JSON itself is valid.
                    if not line.endswith('\n'): raise ValueError(f"Incomplete cache record at {self.cache_path}:{line_number}")
                    try:
                        record = json.loads(line)
                        cache_key = record['cache_key']
                        if not isinstance(cache_key, str):
                            raise ValueError(f"cache_key must be a string:{line_number}: {cache_key}")
                        cache_data = {
                            'coordinates': record['coordinates'],
                            'formatted_address': record['formatted_address'],
                        }
                    except (ValueError, KeyError, TypeError) as e:
                        raise ValueError(f"Invalid cache record at {self.cache_path}:{line_number}: {e}") from e
                    if cache_key in self.geocoding_cache:
                        raise ValueError(f"Duplicate cache key at {self.cache_path}:{line_number}: {cache_key!r}")
                    self.geocoding_cache[cache_key] = cache_data
        self.last_request_time = 0
        self.use_google = use_google and google_api_key is not None
        if self.use_google:
            self.geocode_location = self.geocode_location_google
            self.min_request_interval = 0.1  # 100ms between requests
        else:
            self.geocode_location = self.geocode_location_nominatim
            self._nominatim_lock = Lock()
            self.min_request_interval = 1.0
            self.nominatim_geolocator = Nominatim(user_agent="geo_aot_batch_geocoder")
        
    def update_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Append each new successful result once across this instance's workers."""
        with self._cache_lock:
            # Two workers may both miss the cache before their requests finish.
            if cache_key in self.geocoding_cache:
                return
            cache_data = {
                "coordinates": result["coordinates"],
                "formatted_address": result["formatted_address"],
            }
            record = json.dumps({'cache_key': cache_key, **cache_data}, ensure_ascii=False)
            with self.cache_path.open('a', encoding='utf-8') as f:
                f.write(record + '\n')
            self.geocoding_cache[cache_key] = cache_data

    def geocode_location_google(self, location_description: str) -> Dict[str, Any]:
        """Geocode using Google Geocoding API"""
        result = {
            "input_location": location_description,
            "method": "google_geocoding_api",
            "timestamp": time.time(),
            "success": False,
            "coordinates": None,
            "formatted_address": None,
            "error": None,
            "cached": False
        }
        
        if not self.google_api_key:
            result["error"] = "Google API key not provided"
            return result
            
        # Check cache first
        cache_key = location_description.lower().strip()
        if cache_key in self.geocoding_cache:
            cached_result = self.geocoding_cache[cache_key]
            result.update(cached_result)
            result.update({
                "cached": True,
                "success": True,
            })
            print(f"   ✓ Using cached result for '{location_description}'")
            return result
        
        # Respect rate limits
        remaining = self.min_request_interval - (
            time.monotonic() - self.last_request_time
        )
        if remaining > 0: time.sleep(remaining)
        
        print(f"   → Google Geocoding: '{location_description}'...")
        
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': location_description,
            'key': self.google_api_key
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            self.last_request_time = time.monotonic()
            
            response.raise_for_status()
            data = response.json()
            
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
                    self.update_cache(cache_key, result)
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
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            print(f"   ✗ Unexpected error: {e}")
        
        return result
    
    def geocode_location_nominatim(self, location_description: str) -> Dict[str, Any]:
        """Geocode using Nominatim (OpenStreetMap) - free alternative"""
        result = {
            "input_location": location_description,
            "method": "nominatim_geocoding",
            "timestamp": time.time(),
            "success": False,
            "coordinates": None,
            "formatted_address": None,
            "error": None,
            "cached": False
        }

        cache_key = location_description.lower().strip()
        if cache_key in self.geocoding_cache:
            result.update(self.geocoding_cache[cache_key])
            result.update({
                "cached": True,
                "success": True,
            })
            print(f"   ✓ Using cached result for '{location_description}'")
            return result
        
        print(f"   → Nominatim Geocoding: '{location_description}'...")
        
        try:
            # Serialize calls and measure spacing with a clock unaffected by
            # system time changes. Keep failures inside the rate limit too.
            with self._nominatim_lock:
                remaining = self.min_request_interval - (
                    time.monotonic() - self.last_request_time
                )
                if remaining > 0: time.sleep(remaining)
                try:
                    location = self.nominatim_geolocator.geocode(location_description)
                finally:
                    # Wait a full interval after completion, including failures.
                    self.last_request_time = time.monotonic()
            if location:
                lat, lon = location.latitude, location.longitude
                print(f"   ✓ Found coordinates: ({lat}, {lon})")
                
                # Basic validation for coordinate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    result.update({
                        "success": True,
                        "coordinates": {"latitude": lat, "longitude": lon},
                        "formatted_address": location.address
                    })
                    self.update_cache(cache_key, result)
                else:
                    result["error"] = f"Invalid coordinate ranges: lat={lat}, lon={lon}"
                    print(f"   ✗ Invalid coordinate ranges")
            else:
                result["error"] = "No location found"
                print(f"   ✗ No location found")
        except Exception as e:
            result["error"] = f"Geocoding failed: {str(e)}"
            print(f"   ✗ Geocoding failed: {e}")
        
        return result
    
    def geocode_single_file(self, result_file_path: str) -> Dict[str, Any]:
        """Process a single result file and geocode location descriptions"""
        try:
            with open(result_file_path, 'r') as f:
                data = json.load(f)
            
            file_name = Path(result_file_path).stem
            needs_geocoding = data.get('final_result', {}).get('needs_geocoding', False)
            location_description = data.get('final_result', {}).get('location_description')
            
            if not needs_geocoding or not location_description:
                return {
                    'file_name': file_name,
                    'input_file': result_file_path,
                    'needs_geocoding': needs_geocoding,
                    'geocoded': False,
                    'error': 'No location description found' if needs_geocoding else None
                }
            
            print(f"Processing: {file_name}")
            print(f"Location description: '{location_description}'")
            
            # Try geocoding
            geocoding_result = self.geocode_location(location_description)

            if geocoding_result["success"]:
                # Update the original data with geocoded coordinates
                coords = geocoding_result["coordinates"]
                data['final_result']['pred_coords'] = coords
                data['final_result']['geocoded_coordinates'] = coords
                data['final_result']['geocoding_method'] = geocoding_result["method"]
                data['final_result']['geocoding_timestamp'] = geocoding_result["timestamp"]
                data['final_result']['needs_geocoding'] = False
                
                # Calculate distance error if ground truth available
                ground_truth = data.get('ground_truth') or {}
                gt_coords = ground_truth.get('gt_coords')
                if gt_coords:
                    gt_point = (gt_coords['latitude'], gt_coords['longitude'])
                    pred_point = (coords['latitude'], coords['longitude'])
                    distance_km = round(geodesic(gt_point, pred_point).kilometers, 2)
                    data['ground_truth']['distance_km'] = distance_km
                    print(f"   ✓ Distance error calculated: {distance_km} km")
                
                # Save updated file
                with open(result_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return {
                    'file_name': file_name,
                    'input_file': result_file_path,
                    'needs_geocoding': True,
                    'geocoded': True,
                    'coordinates': coords,
                    'distance_error': ground_truth.get('distance_km'),
                    'geocoding_method': geocoding_result["method"],
                    'formatted_address': geocoding_result.get("formatted_address"),
                    'error': None
                }
            else:
                return {
                    'file_name': file_name,
                    'input_file': result_file_path,
                    'needs_geocoding': True,
                    'geocoded': False,
                    'coordinates': None,
                    'distance_error': None,
                    'geocoding_method': geocoding_result["method"],
                    'error': geocoding_result["error"]
                }
                
        except Exception as e:
            return {
                'file_name': Path(result_file_path).stem,
                'input_file': result_file_path,
                'needs_geocoding': None,
                'geocoded': False,
                'coordinates': None,
                'distance_error': None,
                'error': str(e)
            }
    
    def process_batch_folder(self, batch_folder: str, max_workers: int = 3) -> Dict[str, Any]:
        """Process all result files in a batch folder"""
        batch_path = Path(batch_folder)
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch folder not found: {batch_folder}")
        
        # Find all result files (excluding batch_summary.json)
        result_files = [
            str(f) for f in batch_path.glob("*_results.json")
            if f.name != "batch_summary.json"
        ]
        if not result_files:
            raise ValueError(f"No result files found in: {batch_folder}")
        print(f"Found {len(result_files)} result files for geocoding")

        if not self.use_google: max_workers = 1

        start_time = time.time()
        results = []

        # Process files with limited concurrency to respect API limits
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.geocode_single_file, file_path): file_path
                for file_path in result_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed = len(results)
                    total = len(result_files)
                    print(f"Progress: {completed}/{total} files processed")
                    
                except Exception as e:
                    print(f"Exception processing {file_path}: {e}")
                    results.append({
                        'file_name': Path(file_path).stem,
                        'input_file': file_path,
                        'needs_geocoding': None,
                        'geocoded': False,
                        'coordinates': None,
                        'distance_error': None,
                        'error': str(e)
                    })
        
        # Calculate statistics
        total_time = time.time() - start_time
        needed_geocoding = sum(1 for r in results if r.get('needs_geocoding', False))
        successfully_geocoded = sum(1 for r in results if r.get('geocoded', False))
        # Include processing errors even when the file's geocoding need is unknown.
        failed_results = [
            r for r in results if r.get('error') and not r.get('geocoded')
        ]
        failed_geocoding = len(failed_results)
        
        # Calculate average distance for geocoded results
        geocoded_distances = [
            r['distance_error'] for r in results 
            if r.get('geocoded', False) and r.get('distance_error') is not None
        ]
        avg_distance = sum(geocoded_distances) / len(geocoded_distances) if geocoded_distances else None
        
        # Create geocoding summary
        geocoding_summary = {
            'geocoding_info': {
                'batch_folder': batch_folder,
                'total_files': len(result_files),
                'processing_time': total_time,
                'max_workers': max_workers,
                'use_google_api': self.use_google,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'statistics': {
                'needed_geocoding': needed_geocoding,
                'successfully_geocoded': successfully_geocoded,
                'failed_geocoding': failed_geocoding,
                'geocoding_success_rate': (successfully_geocoded / needed_geocoding * 100) if needed_geocoding > 0 else 0,
                'avg_distance_error_km': avg_distance
            },
            'individual_results': results,
            'failed_geocoding': failed_results
        }
        
        # Save geocoding summary
        summary_file = batch_path / 'geocoding_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(geocoding_summary, f, indent=2)
        
        # Print final summary
        print(f"\n=== Batch Geocoding Complete ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Files processed: {len(result_files)}")
        print(f"Needed geocoding: {needed_geocoding}")
        print(f"Successfully geocoded: {successfully_geocoded}")
        print(f"Failed geocoding: {failed_geocoding}")
        if needed_geocoding > 0:
            print(f"Geocoding success rate: {geocoding_summary['statistics']['geocoding_success_rate']:.1f}%")
        if avg_distance:
            print(f"Average distance error: {avg_distance:.2f} km")
        print(f"Geocoding summary: {summary_file}")
        
        return geocoding_summary

def main():
    parser = argparse.ArgumentParser(description='Batch Geocoding for GeoAoT (Action of Thought) Results')
    parser.add_argument('batch_folder', help='Path to batch results folder')
    parser.add_argument('--google-api-key', help='Google Geocoding API key')
    parser.add_argument('--use-nominatim', action='store_true', 
                        help='Use Nominatim instead of Google (free but slower)')
    parser.add_argument('--max-workers', type=int, default=3,
                        help='Maximum concurrent workers (default: 3)')
    
    args = parser.parse_args()
    
    use_google = not args.use_nominatim and args.google_api_key is not None
    
    if not use_google and not args.use_nominatim:
        print("NOTE: Using Nominatim & setting max_workers=1 to respect Nominatim usage policy.")
        use_google = False
        args.max_workers = 1
    
    geocoder = BatchGeocoder(
        google_api_key=args.google_api_key,
        use_google=use_google,   
    )
    
    try:
        result = geocoder.process_batch_folder(
            args.batch_folder, 
            max_workers=args.max_workers
        )
        
        if result['statistics']['failed_geocoding'] > 0:
            print(f"\nFailed geocoding files:")
            for failed in result['failed_geocoding']:
                print(f"  - {failed['file_name']}: {failed['error']}")
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
