#!/usr/bin/env python3
"""
GPU Evolution Data Visualization Script

Generates comprehensive charts analyzing GPU evolution over time including:
- Transistor count (Moore's Law)
- Performance metrics (TFLOPS)
- Power consumption (TDP)
- Efficiency trends (Performance/Watt)
- Manufacturing process evolution
- Architecture comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

# Configuration
DATA_FILE = Path(__file__).parent.parent / "data" / "gpu_specs.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color schemes for manufacturers
MANUFACTURER_COLORS = {
    'NVIDIA': '#76B900',
    'AMD': '#ED1C24',
    'Intel': '#0071C5',
    '3dfx': '#FFD700',
    'Matrox': '#9932CC',
    'S3 Graphics': '#FF6347'
}

def load_data():
    """Load and preprocess GPU data."""
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✓ Loaded {len(df)} GPU entries")
        
        # Auto-detect and convert dbgpu format to our format
        columns = df.columns.tolist()
        
        # Check if this is dbgpu format (has 'name' instead of 'Model')
        if 'name' in columns and 'Model' not in columns:
            print("  Detected dbgpu format, converting columns...")
            
            # Rename columns
            column_mapping = {
                'name': 'Model',
                'gpu_name': 'Model',
                'manufacturer': 'Manufacturer',
                'architecture': 'Architecture',
                'foundry': 'Foundry',
                'release_date': 'ReleaseDate',
                'transistor_count_m': 'TransistorCount_M',  # Temp name
                'thermal_design_power_w': 'TDP_W',
                'process_size_nm': 'ProcessSize_nm',
                'die_size_mm2': 'DieSize_mm2',
                'single_float_performance_gflop_s': 'TFLOPS_G',  # Temp name
                'memory_bandwidth_gb_s': 'MemoryBandwidth_GBs',
                'base_clock_mhz': 'BaseClock_MHz',
                'boost_clock_mhz': 'BoostClock_MHz',
                'transistor_density_k_mm2': 'TransistorDensity_K',  # Temp name
                'memory_size_gb': 'MemorySize_GB',
                'memory_type': 'MemoryType',
                'shading_units': 'ShadingUnits'
            }
            
            # Rename what we can
            df = df.rename(columns=column_mapping)
            
            # Convert units
            if 'TransistorCount_M' in df.columns:
                df['TransistorCount_B'] = df['TransistorCount_M'] / 1000.0
                df = df.drop('TransistorCount_M', axis=1)
            
            if 'TFLOPS_G' in df.columns:
                df['TFLOPS'] = df['TFLOPS_G'] / 1000.0
                df = df.drop('TFLOPS_G', axis=1)
            
            if 'TransistorDensity_K' in df.columns:
                df['TransistorDensity_M_mm2'] = df['TransistorDensity_K'] / 1000.0
                df = df.drop('TransistorDensity_K', axis=1)
            
            print("  ✓ Converted to visualization format")
        
        # Convert date column (handle both formats)
        date_col = 'ReleaseDate' if 'ReleaseDate' in df.columns else 'release_date'
        if date_col in df.columns:
            df['ReleaseDate'] = pd.to_datetime(df[date_col])
            df['Year'] = df['ReleaseDate'].dt.year
        
        # Calculate derived metrics
        if 'TFLOPS' in df.columns and 'TDP_W' in df.columns:
            df['Performance_per_Watt'] = df['TFLOPS'] / df['TDP_W']
        
        if 'TFLOPS' in df.columns and 'TransistorCount_B' in df.columns:
            df['FLOPS_per_Transistor'] = (df['TFLOPS'] * 1e12) / (df['TransistorCount_B'] * 1e9)
        
        # Sort by date
        df = df.sort_values('ReleaseDate')
        
        return df
    except FileNotFoundError:
        print(f"✗ Error: Could not find {DATA_FILE}")
        print("  Make sure gpu_specs.csv exists in the data/ directory")
        print("  Run: python scripts/collect_data.py --sample 50")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_moores_law_chart(df):
    """Create transistor count over time chart (Moore's Law)."""
    fig = px.scatter(
        df.dropna(subset=['TransistorCount_B']),
        x='Year',
        y='TransistorCount_B',
        color='Manufacturer',
        hover_data=['Model', 'Architecture', 'ProcessSize_nm'],
        title="GPU Transistor Count Evolution (Moore's Law)",
        labels={'TransistorCount_B': 'Transistor Count (Billions)', 'Year': 'Release Year'},
        log_y=True,
        color_discrete_map=MANUFACTURER_COLORS
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    output_file = OUTPUT_DIR / "transistor_count_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_power_consumption_chart(df):
    """Create TDP over time chart."""
    fig = px.scatter(
        df.dropna(subset=['TDP_W']),
        x='Year',
        y='TDP_W',
        color='Manufacturer',
        hover_data=['Model', 'Architecture', 'ProcessSize_nm'],
        title="GPU Power Consumption Evolution (TDP)",
        labels={'TDP_W': 'Thermal Design Power (Watts)', 'Year': 'Release Year'},
        color_discrete_map=MANUFACTURER_COLORS
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    output_file = OUTPUT_DIR / "power_consumption_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_performance_chart(df):
    """Create TFLOPS over time chart."""
    fig = px.scatter(
        df.dropna(subset=['TFLOPS']),
        x='Year',
        y='TFLOPS',
        color='Manufacturer',
        hover_data=['Model', 'Architecture', 'ProcessSize_nm', 'TDP_W'],
        title="GPU Computational Performance Evolution",
        labels={'TFLOPS': 'Performance (TFLOPS)', 'Year': 'Release Year'},
        log_y=True,
        color_discrete_map=MANUFACTURER_COLORS
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    output_file = OUTPUT_DIR / "performance_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_efficiency_chart(df):
    """Create performance per watt over time chart."""
    fig = px.scatter(
        df.dropna(subset=['Performance_per_Watt']),
        x='Year',
        y='Performance_per_Watt',
        color='Manufacturer',
        hover_data=['Model', 'Architecture', 'TFLOPS', 'TDP_W'],
        title="GPU Efficiency Evolution (Performance per Watt)",
        labels={'Performance_per_Watt': 'Efficiency (TFLOPS/Watt)', 'Year': 'Release Year'},
        log_y=True,
        color_discrete_map=MANUFACTURER_COLORS
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    output_file = OUTPUT_DIR / "efficiency_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_process_node_chart(df):
    """Create process size evolution chart."""
    fig = px.scatter(
        df.dropna(subset=['ProcessSize_nm']),
        x='Year',
        y='ProcessSize_nm',
        color='Manufacturer',
        hover_data=['Model', 'Architecture', 'Foundry'],
        title="Manufacturing Process Node Evolution",
        labels={'ProcessSize_nm': 'Process Size (nm)', 'Year': 'Release Year'},
        log_y=True,
        color_discrete_map=MANUFACTURER_COLORS
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    output_file = OUTPUT_DIR / "process_node_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_die_size_chart(df):
    """Create die size over time chart."""
    fig = px.scatter(
        df.dropna(subset=['DieSize_mm2']),
        x='Year',
        y='DieSize_mm2',
        color='Manufacturer',
        hover_data=['Model', 'Architecture', 'ProcessSize_nm', 'TransistorCount_B'],
        title="GPU Die Size Evolution",
        labels={'DieSize_mm2': 'Die Size (mm²)', 'Year': 'Release Year'},
        color_discrete_map=MANUFACTURER_COLORS
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    output_file = OUTPUT_DIR / "die_size_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_memory_bandwidth_chart(df):
    """Create memory bandwidth over time chart."""
    fig = px.scatter(
        df.dropna(subset=['MemoryBandwidth_GBs']),
        x='Year',
        y='MemoryBandwidth_GBs',
        color='Manufacturer',
        hover_data=['Model', 'Architecture', 'MemoryType', 'MemorySize_GB'],
        title="GPU Memory Bandwidth Evolution",
        labels={'MemoryBandwidth_GBs': 'Memory Bandwidth (GB/s)', 'Year': 'Release Year'},
        log_y=True,
        color_discrete_map=MANUFACTURER_COLORS
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    output_file = OUTPUT_DIR / "memory_bandwidth_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_architecture_comparison(df):
    """Create architecture performance comparison chart."""
    # Group by architecture and get median performance
    arch_stats = df.groupby('Architecture').agg({
        'TFLOPS': 'median',
        'TDP_W': 'median',
        'Performance_per_Watt': 'median',
        'Model': 'count'
    }).reset_index()
    arch_stats = arch_stats[arch_stats['Model'] >= 1]  # At least 1 GPU
    arch_stats = arch_stats.sort_values('TFLOPS', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=arch_stats['Architecture'],
        x=arch_stats['TFLOPS'],
        name='Performance (TFLOPS)',
        orientation='h',
        marker_color='#76B900'
    ))
    
    fig.update_layout(
        title="GPU Architecture Performance Comparison (Median TFLOPS)",
        xaxis_title="Performance (TFLOPS)",
        yaxis_title="Architecture",
        template='plotly_white',
        height=max(400, len(arch_stats) * 25)
    )
    
    output_file = OUTPUT_DIR / "architecture_comparison.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def create_multi_metric_dashboard(df):
    """Create comprehensive dashboard with multiple metrics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Transistor Count Evolution',
            'Performance Evolution',
            'Power Consumption',
            'Efficiency (Perf/Watt)'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    manufacturers = df['Manufacturer'].unique()
    
    for mfr in manufacturers:
        mfr_data = df[df['Manufacturer'] == mfr]
        color = MANUFACTURER_COLORS.get(mfr, '#999999')
        
        # Transistor Count
        fig.add_trace(
            go.Scatter(
                x=mfr_data['Year'],
                y=mfr_data['TransistorCount_B'],
                mode='markers',
                name=mfr,
                marker=dict(color=color),
                legendgroup=mfr,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Performance
        fig.add_trace(
            go.Scatter(
                x=mfr_data['Year'],
                y=mfr_data['TFLOPS'],
                mode='markers',
                name=mfr,
                marker=dict(color=color),
                legendgroup=mfr,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Power
        fig.add_trace(
            go.Scatter(
                x=mfr_data['Year'],
                y=mfr_data['TDP_W'],
                mode='markers',
                name=mfr,
                marker=dict(color=color),
                legendgroup=mfr,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Efficiency
        fig.add_trace(
            go.Scatter(
                x=mfr_data['Year'],
                y=mfr_data['Performance_per_Watt'],
                mode='markers',
                name=mfr,
                marker=dict(color=color),
                legendgroup=mfr,
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update axes
    fig.update_yaxes(title_text="Transistor Count (B)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="TFLOPS", type="log", row=1, col=2)
    fig.update_yaxes(title_text="TDP (Watts)", row=2, col=1)
    fig.update_yaxes(title_text="TFLOPS/Watt", type="log", row=2, col=2)
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    
    fig.update_layout(
        title_text="GPU Evolution Dashboard",
        template='plotly_white',
        height=800,
        hovermode='closest'
    )
    
    output_file = OUTPUT_DIR / "dashboard_overview.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    
    return fig

def generate_summary_stats(df):
    """Generate and save summary statistics."""
    stats = {
        'Total GPUs': len(df),
        'Date Range': f"{df['Year'].min()} - {df['Year'].max()}",
        'Manufacturers': df['Manufacturer'].nunique(),
        'Architectures': df['Architecture'].nunique(),
        '\nTransistor Count': '',
        '  Min': f"{df['TransistorCount_B'].min():.2f}B",
        '  Max': f"{df['TransistorCount_B'].max():.2f}B",
        '  Growth Rate': f"{(df['TransistorCount_B'].max() / df['TransistorCount_B'].min()):.0f}x",
        '\nPerformance (TFLOPS)': '',
        '  Min': f"{df['TFLOPS'].min():.2f}",
        '  Max': f"{df['TFLOPS'].max():.2f}",
        '  Growth Rate': f"{(df['TFLOPS'].max() / df['TFLOPS'].min()):.0f}x",
        '\nPower (TDP)': '',
        '  Min': f"{df['TDP_W'].min():.0f}W",
        '  Max': f"{df['TDP_W'].max():.0f}W",
        '\nEfficiency (TFLOPS/W)': '',
        '  Min': f"{df['Performance_per_Watt'].min():.4f}",
        '  Max': f"{df['Performance_per_Watt'].max():.4f}",
        '  Improvement': f"{(df['Performance_per_Watt'].max() / df['Performance_per_Watt'].min()):.0f}x"
    }
    
    output_file = OUTPUT_DIR / "summary_statistics.txt"
    with open(output_file, 'w') as f:
        f.write("GPU Evolution Data - Summary Statistics\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✓ Generated: {output_file.name}")
    
    return stats

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("GPU Evolution Data - Visualization Generator")
    print("="*60 + "\n")
    
    # Load data
    df = load_data()
    
    print(f"\nData Summary:")
    print(f"  - {len(df)} GPU entries")
    print(f"  - {df['Manufacturer'].nunique()} manufacturers")
    print(f"  - {df['Architecture'].nunique()} architectures")
    print(f"  - Date range: {df['Year'].min()} - {df['Year'].max()}")
    
    print(f"\n{'Generating visualizations...'}")
    print("-" * 60)
    
    # Generate all charts
    create_moores_law_chart(df)
    create_power_consumption_chart(df)
    create_performance_chart(df)
    create_efficiency_chart(df)
    create_process_node_chart(df)
    create_die_size_chart(df)
    create_memory_bandwidth_chart(df)
    create_architecture_comparison(df)
    create_multi_metric_dashboard(df)
    
    # Generate summary statistics
    print("-" * 60)
    generate_summary_stats(df)
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    print("Generated files:")
    for file in sorted(OUTPUT_DIR.glob("*.html")):
        print(f"  - {file.name}")
    print(f"  - summary_statistics.txt")
    
    print("\nOpen any .html file in your browser to view interactive charts!")

if __name__ == "__main__":
    main()