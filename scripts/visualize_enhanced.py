#!/usr/bin/env python3
"""
Enhanced GPU Evolution Visualizations

Creates advanced, multi-dimensional charts that tell the full story of GPU evolution:
- Efficiency trends (performance per watt)
- Architectural leaps
- Performance density analysis
- Comparative efficiency scatter plots
- Multi-metric bubble charts
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

# Optional scipy for trend lines
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available, trend lines will be skipped")

# Configuration
DATA_FILE = Path(__file__).parent.parent / "data" / "gpu_specs.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "enhanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Enhanced color schemes
MANUFACTURER_COLORS = {
    'NVIDIA': '#76B900',
    'AMD': '#ED1C24',
    'Intel': '#0071C5',
    'ATI': '#ED1C24',
    '3dfx': '#FFD700',
    'Matrox': '#9932CC',
    'S3 Graphics': '#FF6347',
    'Sony': '#003087',
    'XGI': '#FFA500'
}

def load_data():
    """Load and preprocess GPU data."""
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✓ Loaded {len(df)} GPU entries")
        
        # Auto-detect and convert dbgpu format
        columns = df.columns.tolist()
        if 'name' in columns and 'Model' not in columns:
            print("  Converting dbgpu format...")
            column_mapping = {
                'name': 'Model',
                'gpu_name': 'Model',
                'manufacturer': 'Manufacturer',
                'architecture': 'Architecture',
                'foundry': 'Foundry',
                'release_date': 'ReleaseDate',
                'transistor_count_m': 'TransistorCount_M',
                'thermal_design_power_w': 'TDP_W',
                'process_size_nm': 'ProcessSize_nm',
                'die_size_mm2': 'DieSize_mm2',
                'single_float_performance_gflop_s': 'TFLOPS_G',
                'memory_bandwidth_gb_s': 'MemoryBandwidth_GBs',
                'base_clock_mhz': 'BaseClock_MHz',
                'boost_clock_mhz': 'BoostClock_MHz',
                'transistor_density_k_mm2': 'TransistorDensity_K',
                'memory_size_gb': 'MemorySize_GB',
                'memory_type': 'MemoryType',
                'shading_units': 'ShadingUnits'
            }
            df = df.rename(columns=column_mapping)
            
            if 'TransistorCount_M' in df.columns:
                df['TransistorCount_B'] = df['TransistorCount_M'] / 1000.0
            if 'TFLOPS_G' in df.columns:
                df['TFLOPS'] = df['TFLOPS_G'] / 1000.0
            if 'TransistorDensity_K' in df.columns:
                df['TransistorDensity_M_mm2'] = df['TransistorDensity_K'] / 1000.0
        
        # Convert dates
        date_col = 'ReleaseDate' if 'ReleaseDate' in df.columns else 'release_date'
        if date_col in df.columns:
            df['ReleaseDate'] = pd.to_datetime(df[date_col])
            df['Year'] = df['ReleaseDate'].dt.year
        
        # Calculate derived metrics
        df['Performance_per_Watt'] = df['TFLOPS'] / df['TDP_W']
        df['FLOPS_per_Transistor'] = (df['TFLOPS'] * 1e12) / (df['TransistorCount_B'] * 1e9)
        df['Performance_Density'] = df['TFLOPS'] / df['DieSize_mm2']  # TFLOPS per mm²
        df['Transistors_per_Watt'] = df['TransistorCount_B'] / df['TDP_W']  # Billions per watt
        
        # Sort by date
        df = df.sort_values('ReleaseDate')
        
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_efficiency_story(df):
    """Create the efficiency evolution story - THE KEY METRIC"""
    # Filter to recent years for cleaner view
    df_modern = df[df['Year'] >= 2010].dropna(subset=['Performance_per_Watt'])
    
    fig = go.Figure()
    
    for mfr in df_modern['Manufacturer'].unique():
        mfr_data = df_modern[df_modern['Manufacturer'] == mfr]
        
        fig.add_trace(go.Scatter(
            x=mfr_data['Year'],
            y=mfr_data['Performance_per_Watt'],
            mode='markers',
            name=mfr,
            marker=dict(
                size=10,
                color=MANUFACTURER_COLORS.get(mfr, '#999999'),
                line=dict(width=1, color='white')
            ),
            text=mfr_data['Model'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Year: %{x}<br>' +
                         'Efficiency: %{y:.3f} TFLOPS/W<br>' +
                         '<extra></extra>'
        ))
    
    # Add exponential trend line (if scipy available)
    if SCIPY_AVAILABLE:
        valid_data = df_modern.dropna(subset=['Year', 'Performance_per_Watt'])
        if len(valid_data) > 10:
            z = np.polyfit(valid_data['Year'], np.log(valid_data['Performance_per_Watt']), 1)
            p = np.poly1d(z)
            years_range = np.linspace(valid_data['Year'].min(), valid_data['Year'].max(), 100)
            trend = np.exp(p(years_range))
            
            fig.add_trace(go.Scatter(
                x=years_range,
                y=trend,
                mode='lines',
                name='Exponential Trend',
                line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dash'),
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title={
            'text': 'GPU Efficiency Revolution: Performance per Watt (2010-Present)<br>' +
                   '<sub>The Real Story of GPU Evolution</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Release Year',
        yaxis_title='Efficiency (TFLOPS per Watt)',
        yaxis_type='log',
        template='plotly_white',
        hovermode='closest',
        height=700,
        legend=dict(x=0.02, y=0.98)
    )
    
    output_file = OUTPUT_DIR / "efficiency_revolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    return fig

def create_power_vs_performance(df):
    """Scatter: TDP vs Performance - shows efficiency clusters"""
    df_clean = df.dropna(subset=['TDP_W', 'TFLOPS', 'Year', 'DieSize_mm2'])  # Added DieSize_mm2
    df_recent = df_clean[df_clean['Year'] >= 2015]  # Focus on modern GPUs
    
    # Additional safety check
    df_recent = df_recent[df_recent['DieSize_mm2'] > 0]  # Remove zero/negative values
    
    if len(df_recent) == 0:
        print("⚠ Warning: No data available for power_vs_performance chart")
        return None
    
    fig = px.scatter(
        df_recent,
        x='TDP_W',
        y='TFLOPS',
        color='Manufacturer',
        size='DieSize_mm2',
        hover_data=['Model', 'Year', 'Architecture', 'Performance_per_Watt'],
        title='Power vs Performance: The Efficiency Landscape (2015+)<br>' +
              '<sub>Size = Die Area | Diagonal = Efficiency</sub>',
        labels={
            'TDP_W': 'Thermal Design Power (Watts)',
            'TFLOPS': 'Performance (TFLOPS)'
        },
        color_discrete_map=MANUFACTURER_COLORS,
        log_y=True
    )
    
    # Add diagonal efficiency lines
    tdp_range = np.linspace(df_recent['TDP_W'].min(), df_recent['TDP_W'].max(), 100)
    
    for efficiency in [0.01, 0.05, 0.1, 0.2, 0.5]:
        fig.add_trace(go.Scatter(
            x=tdp_range,
            y=tdp_range * efficiency,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.1)', width=1, dash='dash'),
            name=f'{efficiency:.2f} TFLOPS/W',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add label
        fig.add_annotation(
            x=tdp_range[-1],
            y=tdp_range[-1] * efficiency,
            text=f'{efficiency:.2f} TFLOPS/W',
            showarrow=False,
            xanchor='left',
            font=dict(size=9, color='rgba(0,0,0,0.4)')
        )
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        hovermode='closest'
    )
    
    output_file = OUTPUT_DIR / "power_vs_performance.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    return fig

def create_bubble_chart(df):
    """Ultimate bubble chart: Year vs Performance, size=die, color=efficiency"""
    df_clean = df.dropna(subset=['Year', 'TFLOPS', 'DieSize_mm2', 'Performance_per_Watt'])
    df_modern = df_clean[df_clean['Year'] >= 2010]
    
    # Remove invalid values
    df_modern = df_modern[df_modern['DieSize_mm2'] > 0]
    df_modern = df_modern[df_modern['Performance_per_Watt'] > 0]
    
    if len(df_modern) == 0:
        print("⚠ Warning: No data available for bubble chart")
        return None
    
    # Normalize die size for bubble sizing
    df_modern['BubbleSize'] = df_modern['DieSize_mm2'] / 10  # Scale for visibility
    
    fig = px.scatter(
        df_modern,
        x='Year',
        y='TFLOPS',
        size='BubbleSize',
        color='Performance_per_Watt',
        hover_data=['Model', 'Manufacturer', 'Architecture', 'TDP_W', 'DieSize_mm2'],
        title='GPU Evolution: The Complete Picture<br>' +
              '<sub>Y-axis: Performance | Bubble Size: Die Area | Color: Efficiency</sub>',
        labels={
            'Year': 'Release Year',
            'TFLOPS': 'Performance (TFLOPS)',
            'Performance_per_Watt': 'Efficiency (TFLOPS/W)'
        },
        color_continuous_scale='Viridis',
        log_y=True
    )
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        hovermode='closest',
        coloraxis_colorbar=dict(
            title="Efficiency<br>(TFLOPS/W)",
            tickformat='.3f'
        )
    )
    
    output_file = OUTPUT_DIR / "bubble_complete_story.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    return fig

def create_architecture_evolution(df):
    """Show generational leaps by architecture"""
    df_clean = df.dropna(subset=['Architecture', 'Performance_per_Watt', 'Year'])
    df_modern = df_clean[df_clean['Year'] >= 2010]
    
    # Get top architectures by GPU count
    top_archs = df_modern['Architecture'].value_counts().head(20).index.tolist()
    df_top = df_modern[df_modern['Architecture'].isin(top_archs)]
    
    # Calculate median efficiency per architecture
    arch_efficiency = df_top.groupby(['Architecture', 'Manufacturer']).agg({
        'Performance_per_Watt': 'median',
        'Year': 'median',
        'Model': 'count'
    }).reset_index()
    
    arch_efficiency = arch_efficiency.sort_values('Year')
    
    fig = px.scatter(
        arch_efficiency,
        x='Year',
        y='Performance_per_Watt',
        color='Manufacturer',
        size='Model',
        text='Architecture',
        title='GPU Architecture Evolution: Efficiency by Generation<br>' +
              '<sub>Each point = GPU architecture median | Size = number of GPUs</sub>',
        labels={
            'Year': 'Architecture Release Year',
            'Performance_per_Watt': 'Median Efficiency (TFLOPS/W)',
            'Model': 'GPU Count'
        },
        color_discrete_map=MANUFACTURER_COLORS,
        log_y=True
    )
    
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=8)
    )
    
    fig.update_layout(
        template='plotly_white',
        height=800,
        hovermode='closest'
    )
    
    output_file = OUTPUT_DIR / "architecture_generations.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    return fig

def create_triple_evolution(df):
    """Three metrics over time: Transistors, Performance, Power"""
    df_clean = df.dropna(subset=['Year', 'TransistorCount_B', 'TFLOPS', 'TDP_W'])
    
    # Normalize to starting point
    min_year = df_clean['Year'].min()
    base_year_data = df_clean[df_clean['Year'] == min_year]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Transistor Count Growth',
            'Performance Growth', 
            'Power Consumption Growth'
        ),
        vertical_spacing=0.08
    )
    
    for mfr in df_clean['Manufacturer'].unique():
        mfr_data = df_clean[df_clean['Manufacturer'] == mfr]
        color = MANUFACTURER_COLORS.get(mfr, '#999999')
        
        fig.add_trace(
            go.Scatter(
                x=mfr_data['Year'],
                y=mfr_data['TransistorCount_B'],
                mode='markers',
                name=mfr,
                marker=dict(color=color, size=6),
                legendgroup=mfr,
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=mfr_data['Year'],
                y=mfr_data['TFLOPS'],
                mode='markers',
                name=mfr,
                marker=dict(color=color, size=6),
                legendgroup=mfr,
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=mfr_data['Year'],
                y=mfr_data['TDP_W'],
                mode='markers',
                name=mfr,
                marker=dict(color=color, size=6),
                legendgroup=mfr,
                showlegend=False
            ),
            row=3, col=1
        )
    
    fig.update_yaxes(title_text="Transistors (B)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="TFLOPS", type="log", row=2, col=1)
    fig.update_yaxes(title_text="TDP (Watts)", row=3, col=1)
    fig.update_xaxes(title_text="Year", row=3, col=1)
    
    fig.update_layout(
        title_text="GPU Evolution: Three Key Metrics",
        template='plotly_white',
        height=900,
        hovermode='closest'
    )
    
    output_file = OUTPUT_DIR / "triple_metric_evolution.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    return fig

def create_efficiency_heatmap(df):
    """Heatmap showing efficiency by year and manufacturer"""
    df_clean = df.dropna(subset=['Year', 'Manufacturer', 'Performance_per_Watt'])
    df_modern = df_clean[df_clean['Year'] >= 2010]
    
    # Create pivot table
    pivot = df_modern.pivot_table(
        values='Performance_per_Watt',
        index='Manufacturer',
        columns='Year',
        aggfunc='median'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        text=np.round(pivot.values, 3),
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        colorbar=dict(title="TFLOPS/W")
    ))
    
    fig.update_layout(
        title='GPU Efficiency Heatmap: Median by Year & Manufacturer<br>' +
              '<sub>Darker = More Efficient</sub>',
        xaxis_title='Year',
        yaxis_title='Manufacturer',
        template='plotly_white',
        height=500
    )
    
    output_file = OUTPUT_DIR / "efficiency_heatmap.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    return fig

def create_top_performers_comparison(df):
    """Compare the top performing GPUs across different metrics"""
    df_clean = df.dropna(subset=['TFLOPS', 'Performance_per_Watt', 'TransistorCount_B'])
    df_modern = df_clean[df_clean['Year'] >= 2015]
    
    # Get top 20 by raw performance
    top_perf = df_modern.nlargest(20, 'TFLOPS')
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Performance', 'Efficiency', 'Transistor Count'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Sort each metric separately
    perf_sorted = top_perf.sort_values('TFLOPS', ascending=True)
    eff_sorted = top_perf.sort_values('Performance_per_Watt', ascending=True)
    trans_sorted = top_perf.sort_values('TransistorCount_B', ascending=True)
    
    fig.add_trace(
        go.Bar(
            y=perf_sorted['Model'],
            x=perf_sorted['TFLOPS'],
            orientation='h',
            marker_color='#76B900',
            name='TFLOPS'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            y=eff_sorted['Model'],
            x=eff_sorted['Performance_per_Watt'],
            orientation='h',
            marker_color='#ED1C24',
            name='TFLOPS/W'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            y=trans_sorted['Model'],
            x=trans_sorted['TransistorCount_B'],
            orientation='h',
            marker_color='#0071C5',
            name='Billions'
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="TFLOPS", row=1, col=1)
    fig.update_xaxes(title_text="TFLOPS/W", row=1, col=2)
    fig.update_xaxes(title_text="Transistors (B)", row=1, col=3)
    
    fig.update_layout(
        title_text="Top 20 Modern GPUs: Three Perspectives (2015+)",
        template='plotly_white',
        height=800,
        showlegend=False
    )
    
    output_file = OUTPUT_DIR / "top_performers_comparison.html"
    fig.write_html(str(output_file))
    print(f"✓ Generated: {output_file.name}")
    return fig

def main():
    """Generate all enhanced visualizations"""
    print("\n" + "="*60)
    print("Enhanced GPU Visualization Generator")
    print("="*60 + "\n")
    
    df = load_data()
    
    print(f"\nData Summary:")
    print(f"  - {len(df)} GPU entries")
    print(f"  - {df['Manufacturer'].nunique()} manufacturers")
    print(f"  - Date range: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
    
    print(f"\nGenerating enhanced visualizations...")
    print("-" * 60)
    
    # Generate enhanced charts with error handling
    charts_generated = 0
    charts_failed = 0
    
    try:
        create_efficiency_story(df)
        charts_generated += 1
    except Exception as e:
        print(f"✗ Failed: efficiency_revolution.html - {e}")
        charts_failed += 1
    
    try:
        create_power_vs_performance(df)
        charts_generated += 1
    except Exception as e:
        print(f"✗ Failed: power_vs_performance.html - {e}")
        charts_failed += 1
    
    try:
        create_bubble_chart(df)
        charts_generated += 1
    except Exception as e:
        print(f"✗ Failed: bubble_complete_story.html - {e}")
        charts_failed += 1
    
    try:
        create_architecture_evolution(df)
        charts_generated += 1
    except Exception as e:
        print(f"✗ Failed: architecture_generations.html - {e}")
        charts_failed += 1
    
    try:
        create_triple_evolution(df)
        charts_generated += 1
    except Exception as e:
        print(f"✗ Failed: triple_metric_evolution.html - {e}")
        charts_failed += 1
    
    try:
        create_efficiency_heatmap(df)
        charts_generated += 1
    except Exception as e:
        print(f"✗ Failed: efficiency_heatmap.html - {e}")
        charts_failed += 1
    
    try:
        create_top_performers_comparison(df)
        charts_generated += 1
    except Exception as e:
        print(f"✗ Failed: top_performers_comparison.html - {e}")
        charts_failed += 1
    
    print("\n" + "="*60)
    print(f"✓ Generated {charts_generated} charts successfully")
    if charts_failed > 0:
        print(f"✗ {charts_failed} charts failed (likely due to missing data)")
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    print("Generated enhanced charts:")
    for file in sorted(OUTPUT_DIR.glob("*.html")):
        print(f"  - {file.name}")
    
    if charts_generated > 0:
        print("\n🎯 KEY INSIGHT CHARTS:")
        print("  1. efficiency_revolution.html - THE main story")
        print("  2. power_vs_performance.html - Efficiency landscape")
        print("  3. bubble_complete_story.html - Everything in one view")

if __name__ == "__main__":
    main()