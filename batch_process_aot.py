import hydra
from omegaconf import DictConfig
from geo_aot_geoguess import GeoAoTGuesser
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Any

def process_single_file(json_file_path: str, pano_folder: str, output_dir: Path, 
                       ai_config: DictConfig, ai_keys: DictConfig, 
                       max_steps: int, debug: bool) -> Dict[str, Any]:
    """Process a single JSON file"""
    try:
        start_time = time.time()
        file_name = Path(json_file_path).stem
        
        # Initialize system
        guesser = GeoAoTGuesser(
            json_file_path, 
            pano_folder,
            ai_config=ai_config,
            ai_keys=ai_keys,
            max_steps=max_steps,
            debug=debug
        )
        
        # Run session (ground truth will be auto-extracted from JSON)
        result = guesser.run_geo_aot_session(interactive=False)
        
        # Save results to output folder with matching filename
        output_file = output_dir / f'{file_name}_results.json'
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
            'steps_taken': result['geo_aot']['steps_taken'],
            'max_steps': result['geo_aot']['max_steps'],
            'final_coordinates': result['final_result']['pred_coords'],
            'location_description': result['final_result'].get('location_description'),
            'needs_geocoding': result['final_result'].get('needs_geocoding', False),
            'distance_error': result.get('ground_truth', {}).get('distance_km'),
            'error': None
        }
        
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
            'steps_taken': 0,
            'max_steps': max_steps,
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
    """Main batch processing function"""
    print("=== GeoAoT (Action of Thought) Batch Processing System ===")
    print(f"Config: {cfg}")
    
    # Get configuration parameters
    input_folder = cfg.batch_process.input_graphs_folder
    pano_folder = cfg.batch_process.pano_folder
    max_steps = cfg.max_steps
    debug = cfg.debug
    max_workers = cfg.batch_process.get('max_workers', 4)  # Default to 4 concurrent workers
    cfg.output_folder = os.path.join(cfg.output_folder, cfg.ai_config.model)
    
    # Create output directory
    output_dir = Path(cfg.output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Get all JSON files to process
    try:
        json_files = get_json_files(input_folder)
        print(f"Found {len(json_files)} JSON files to process")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # Process files concurrently
    start_time = time.time()
    results = []
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Starting batch processing with {max_workers} concurrent workers...")
        
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file, 
                json_file, pano_folder, output_dir,
                cfg.ai_config, cfg.ai_keys, max_steps, debug
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
                    'steps_taken': 0,
                    'max_steps': max_steps,
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
    
    total_steps = sum(r['steps_taken'] for r in results if r['success'])
    avg_steps = total_steps / successful if successful > 0 else 0
    
    successful_distances = [r['distance_error'] for r in results if r['success'] and r['distance_error'] is not None]
    avg_distance = sum(successful_distances) / len(successful_distances) if successful_distances else None
    
    # Geocoding statistics
    needs_geocoding = sum(1 for r in results if r['success'] and r.get('needs_geocoding', False))
    has_coordinates = sum(1 for r in results if r['success'] and r['final_coordinates'] is not None)
    has_descriptions = sum(1 for r in results if r['success'] and r.get('location_description') is not None)
    
    # Create batch summary
    batch_summary = {
        'batch_info': {
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
            'avg_steps_per_file': avg_steps,
            'avg_distance_error_km': avg_distance,
            'total_steps_taken': total_steps,
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
    summary_file = output_dir / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Print final summary
    print(f"\n=== Batch Processing Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Files processed: {len(json_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {batch_summary['statistics']['success_rate']:.1f}%")
    print(f"Average steps per file: {avg_steps:.1f}")
    
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
