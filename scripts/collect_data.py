#!/usr/bin/env python3
"""
GPU Data Collection Script using dbgpu

This script uses the dbgpu library to fetch GPU specifications from TechPowerUp
and populate the gpu_specs.csv file with the data needed for analysis.

Usage:
    python scripts/collect_data.py --manufacturer NVIDIA AMD --start-year 2010
    python scripts/collect_data.py --all  # Collect all GPUs (will take a while!)
    python scripts/collect_data.py --sample 100  # Collect a random sample
"""

import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import sys

try:
    from dbgpu import GPUDatabase
except ImportError:
    print("✗ Error: dbgpu is not installed.")
    print("  Install it with: pip install dbgpu")
    print("  Or install all dependencies: pip install -r scripts/requirements.txt")
    sys.exit(1)

# File paths
DATA_FILE = Path(__file__).parent.parent / "data" / "gpu_specs.csv"

# CSV columns matching our data dictionary
CSV_COLUMNS = [
    'Model',
    'Manufacturer',
    'Architecture',
    'Foundry',
    'ReleaseDate',
    'TransistorCount_B',
    'TDP_W',
    'ProcessSize_nm',
    'DieSize_mm2',
    'TFLOPS',
    'MemoryBandwidth_GBs',
    'BaseClock_MHz',
    'BoostClock_MHz',
    'TransistorDensity_M_mm2',
    'MemorySize_GB',
    'MemoryType',
    'ShadingUnits'
]

def convert_gpu_to_row(spec) -> dict:
    """
    Convert a dbgpu GPUSpecification to our CSV row format.
    
    dbgpu uses different units and field names than our schema:
    - transistor_count_m (millions) -> TransistorCount_B (billions)
    - single_float_performance_gflop_s (GFLOPS) -> TFLOPS (TFLOPS)
    - transistor_density_k_mm2 (thousands/mm²) -> TransistorDensity_M_mm2 (millions/mm²)
    """
    
    # Convert transistor count from millions to billions
    transistor_count_b = None
    if spec.transistor_count_m is not None:
        transistor_count_b = spec.transistor_count_m / 1000.0
    
    # Convert GFLOPS to TFLOPS
    tflops = None
    if spec.single_float_performance_gflop_s is not None:
        tflops = spec.single_float_performance_gflop_s / 1000.0
    
    # Convert transistor density from thousands/mm² to millions/mm²
    transistor_density_m_mm2 = None
    if spec.transistor_density_k_mm2 is not None:
        transistor_density_m_mm2 = spec.transistor_density_k_mm2 / 1000.0
    
    # Format release date
    release_date = None
    if spec.release_date is not None:
        release_date = spec.release_date.strftime('%Y-%m-%d')
    
    return {
        'Model': spec.gpu_name or spec.name,
        'Manufacturer': spec.manufacturer,
        'Architecture': spec.architecture or '',
        'Foundry': spec.foundry or '',
        'ReleaseDate': release_date or '',
        'TransistorCount_B': transistor_count_b or '',
        'TDP_W': spec.thermal_design_power_w or '',
        'ProcessSize_nm': spec.process_size_nm or '',
        'DieSize_mm2': spec.die_size_mm2 or '',
        'TFLOPS': tflops or '',
        'MemoryBandwidth_GBs': spec.memory_bandwidth_gb_s or '',
        'BaseClock_MHz': spec.base_clock_mhz or '',
        'BoostClock_MHz': spec.boost_clock_mhz or '',
        'TransistorDensity_M_mm2': transistor_density_m_mm2 or '',
        'MemorySize_GB': spec.memory_size_gb or '',
        'MemoryType': spec.memory_type or '',
        'ShadingUnits': spec.shading_units or ''
    }

def load_existing_gpus() -> set:
    """Load the set of GPUs already in the CSV to avoid duplicates."""
    existing = set()
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Use Model + Manufacturer as unique key
                    key = (row['Model'], row['Manufacturer'])
                    existing.add(key)
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")
    return existing

def filter_gpus(database: GPUDatabase, 
                manufacturers: Optional[List[str]] = None,
                start_year: Optional[int] = None,
                end_year: Optional[int] = None,
                architectures: Optional[List[str]] = None,
                sample_size: Optional[int] = None) -> List:
    """Filter GPUs based on criteria."""
    
    all_gpus = list(database.specifications.values())
    filtered = []
    
    for gpu in all_gpus:
        # Filter by manufacturer
        if manufacturers and gpu.manufacturer not in manufacturers:
            continue
        
        # Filter by year
        if gpu.release_date:
            year = gpu.release_date.year
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue
        elif start_year or end_year:
            # Skip GPUs without release date if filtering by year
            continue
        
        # Filter by architecture
        if architectures and gpu.architecture not in architectures:
            continue
        
        filtered.append(gpu)
    
    # Random sample if requested
    if sample_size and len(filtered) > sample_size:
        import random
        filtered = random.sample(filtered, sample_size)
    
    return filtered

def collect_data(manufacturers: Optional[List[str]] = None,
                start_year: Optional[int] = None,
                end_year: Optional[int] = None,
                architectures: Optional[List[str]] = None,
                sample_size: Optional[int] = None,
                append: bool = True,
                verbose: bool = True):
    """
    Collect GPU data from dbgpu and save to CSV.
    
    Args:
        manufacturers: List of manufacturers to include (e.g., ['NVIDIA', 'AMD'])
        start_year: Minimum release year
        end_year: Maximum release year
        architectures: List of architectures to include
        sample_size: Random sample size (for testing)
        append: If True, append to existing CSV; if False, overwrite
        verbose: Print progress messages
    """
    
    if verbose:
        print("\n" + "="*60)
        print("GPU Data Collection - dbgpu to CSV")
        print("="*60 + "\n")
    
    # Load database
    if verbose:
        print("Loading dbgpu database...")
    database = GPUDatabase.default()
    if verbose:
        print(f"✓ Loaded database with {len(database.specifications)} GPUs\n")
    
    # Load existing GPUs if appending
    existing_gpus = set()
    if append and DATA_FILE.exists():
        existing_gpus = load_existing_gpus()
        if verbose:
            print(f"Found {len(existing_gpus)} existing GPUs in CSV")
    
    # Filter GPUs
    if verbose:
        print("\nFiltering GPUs...")
        if manufacturers:
            print(f"  - Manufacturers: {', '.join(manufacturers)}")
        if start_year:
            print(f"  - Start year: {start_year}")
        if end_year:
            print(f"  - End year: {end_year}")
        if architectures:
            print(f"  - Architectures: {', '.join(architectures)}")
        if sample_size:
            print(f"  - Sample size: {sample_size}")
    
    filtered_gpus = filter_gpus(
        database,
        manufacturers=manufacturers,
        start_year=start_year,
        end_year=end_year,
        architectures=architectures,
        sample_size=sample_size
    )
    
    if verbose:
        print(f"✓ Found {len(filtered_gpus)} matching GPUs")
    
    # Convert to rows and filter duplicates
    new_rows = []
    skipped = 0
    for gpu in filtered_gpus:
        key = (gpu.gpu_name or gpu.name, gpu.manufacturer)
        if key not in existing_gpus:
            row = convert_gpu_to_row(gpu)
            new_rows.append(row)
        else:
            skipped += 1
    
    if verbose:
        print(f"✓ {len(new_rows)} new GPUs to add")
        if skipped > 0:
            print(f"  ({skipped} duplicates skipped)")
    
    if not new_rows:
        if verbose:
            print("\nNo new data to add!")
        return 0
    
    # Write to CSV
    if verbose:
        print(f"\nWriting to {DATA_FILE}...")
    
    # Determine mode and whether to write header
    file_exists = DATA_FILE.exists()
    mode = 'a' if (append and file_exists) else 'w'
    write_header = not (append and file_exists)  # Write header if new file or overwriting
    
    with open(DATA_FILE, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        
        if write_header:
            writer.writeheader()
            if verbose:
                print(f"  ✓ Wrote CSV header row")
        
        writer.writerows(new_rows)
    
    if verbose:
        print(f"✓ Successfully added {len(new_rows)} GPUs to {DATA_FILE}")
        print("\n" + "="*60)
        print("Collection complete!")
        print("="*60 + "\n")
        print(f"Run 'python scripts/visualize.py' to generate charts.")
    
    return len(new_rows)

def main():
    parser = argparse.ArgumentParser(
        description='Collect GPU data from dbgpu and save to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all NVIDIA GPUs from 2015 onwards
  python scripts/collect_data.py --manufacturer NVIDIA --start-year 2015
  
  # Collect both NVIDIA and AMD GPUs
  python scripts/collect_data.py --manufacturer NVIDIA AMD
  
  # Collect a random sample of 100 GPUs for testing
  python scripts/collect_data.py --sample 100
  
  # Collect all GPUs (warning: this will take a while!)
  python scripts/collect_data.py --all
  
  # Overwrite existing CSV instead of appending
  python scripts/collect_data.py --manufacturer NVIDIA --overwrite
        """
    )
    
    parser.add_argument(
        '--manufacturer', '-m',
        nargs='+',
        help='GPU manufacturers to include (e.g., NVIDIA AMD Intel)'
    )
    
    parser.add_argument(
        '--start-year', '-y',
        type=int,
        help='Minimum release year (e.g., 2010)'
    )
    
    parser.add_argument(
        '--end-year', '-Y',
        type=int,
        help='Maximum release year (e.g., 2024)'
    )
    
    parser.add_argument(
        '--architecture', '-a',
        nargs='+',
        help='GPU architectures to include (e.g., Pascal Ampere)'
    )
    
    parser.add_argument(
        '--sample', '-s',
        type=int,
        help='Collect a random sample of N GPUs (useful for testing)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Collect all GPUs in the database (no filtering)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing CSV instead of appending'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.manufacturer, args.start_year, args.end_year, 
                args.architecture, args.sample, args.all]):
        print("Error: Must specify at least one filter option or --all")
        print("Run with --help for usage examples")
        sys.exit(1)
    
    # Check if we need to create/overwrite the CSV
    if args.overwrite and DATA_FILE.exists():
        print(f"\n⚠ Warning: --overwrite will replace existing CSV file")
        print(f"  Current file has data, it will be deleted.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Collect data
    try:
        count = collect_data(
            manufacturers=args.manufacturer,
            start_year=args.start_year,
            end_year=args.end_year,
            architectures=args.architecture,
            sample_size=args.sample,
            append=not args.overwrite,
            verbose=not args.quiet
        )
        sys.exit(0 if count > 0 else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()