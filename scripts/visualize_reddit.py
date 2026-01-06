#!/usr/bin/env python3
"""
GPU Transistor Count Visualization for Reddit

Creates a clean, linear-scale chart showing GPU transistor count evolution
optimized for r/dataisbeautiful posting.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

# Configuration
DATA_FILE = Path(__file__).parent.parent / "data" / "gpu_specs.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "reddit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
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
                'release_date': 'ReleaseDate',
                'transistor_count_m': 'TransistorCount_M',
            }
            df = df.rename(columns=column_mapping)
            
            if 'TransistorCount_M' in df.columns:
                df['TransistorCount_B'] = df['TransistorCount_M'] / 1000.0
        
        # Convert dates
        date_col = 'ReleaseDate' if 'ReleaseDate' in df.columns else 'release_date'
        if date_col in df.columns:
            df['ReleaseDate'] = pd.to_datetime(df[date_col])
            df['Year'] = df['ReleaseDate'].dt.year
        
        df = df.sort_values('ReleaseDate')
        return df
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_reddit_transistor_chart(df):
    """
    Create THE Reddit chart: Transistor count over time (LINEAR SCALE)
    """
    # Filter data
    df_clean = df.dropna(subset=['Year', 'TransistorCount_B'])
    df_filtered = df_clean[df_clean['Year'] >= 1995]  # Start from 1995
    
    print(f"\nChart data: {len(df_filtered)} GPUs from {df_filtered['Year'].min():.0f}-{df_filtered['Year'].max():.0f}")
    
    fig = go.Figure()
    
    # For each manufacturer, show only the maximum transistor count per year (flagship GPUs)
    for mfr in df_filtered['Manufacturer'].unique():
        mfr_data = df_filtered[df_filtered['Manufacturer'] == mfr]
        
        # Get the maximum transistor count for each year
        flagship = mfr_data.loc[mfr_data.groupby('Year')['TransistorCount_B'].idxmax()]
        flagship = flagship.sort_values('Year')
        
        fig.add_trace(go.Scatter(
            x=flagship['Year'],
            y=flagship['TransistorCount_B'],
            mode='lines+markers',
            name=mfr,
            line=dict(
                color=MANUFACTURER_COLORS.get(mfr, '#999999'),
                width=2.5
            ),
            marker=dict(
                size=6,
                color=MANUFACTURER_COLORS.get(mfr, '#999999'),
                line=dict(width=1, color='white')
            ),
            text=flagship['Model'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Year: %{x}<br>' +
                         'Transistors: %{y:.1f}B<br>' +
                         '<extra></extra>'
        ))
    
    # Add average trend line (dotted) - using all GPUs for average
    yearly_avg = df_filtered.groupby('Year')['TransistorCount_B'].mean().reset_index()
    yearly_avg = yearly_avg.sort_values('Year')
    
    fig.add_trace(go.Scatter(
        x=yearly_avg['Year'],
        y=yearly_avg['TransistorCount_B'],
        mode='lines',
        name='Average (All GPUs)',
        line=dict(color='rgba(0,0,0,0.6)', width=3, dash='dot'),
        hovertemplate='<b>Average</b><br>' +
                     'Year: %{x}<br>' +
                     'Transistors: %{y:.1f}B<br>' +
                     '<extra></extra>'
    ))
    
    # Add exponential trend line (dashed)
    x_data = df_filtered['Year'].values
    y_data = df_filtered['TransistorCount_B'].values
    
    mask = y_data > 0
    x_trend = x_data[mask]
    y_trend = y_data[mask]
    
    if len(x_trend) > 10:
        log_y = np.log(y_trend)
        z = np.polyfit(x_trend, log_y, 1)
        
        x_smooth = np.linspace(x_trend.min(), x_trend.max(), 200)
        y_smooth = np.exp(z[1] + z[0] * x_smooth)
        
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name='Exponential Trend',
            line=dict(color='rgba(0,0,0,0.4)', width=3, dash='dash'),
            hovertemplate='<b>Exponential Trend</b><br>' +
                         'Year: %{x}<br>' +
                         'Projected: %{y:.1f}B<br>' +
                         '<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title={
            'text': 'GPU Transistor Count (1995-2025)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2C3E50'}
        },
        xaxis=dict(
            title=dict(
                text='Year',
                font=dict(size=16)
            ),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickfont={'size': 12}
        ),
        yaxis=dict(
            title=dict(
                text='Transistor Count (Billions)',
                font=dict(size=16)
            ),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickfont={'size': 12},
            # LINEAR SCALE - this is the key!
            type='linear'
        ),
        template='plotly_white',
        hovermode='closest',
        height=600,
        width=1000,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save as HTML
    output_html = OUTPUT_DIR / "transistor_count_reddit.html"
    fig.write_html(str(output_html))
    print(f"✓ Generated: {output_html}")
    
    # Save as static PNG (better for Reddit)
    output_png = OUTPUT_DIR / "transistor_count_reddit.png"
    try:
        import kaleido
        fig.write_image(
            str(output_png), 
            width=1200, 
            height=700, 
            scale=2,
            format='png'
        )
        print(f"✓ Generated: {output_png}")
        print(f"  Image size: {output_png.stat().st_size / 1024:.1f} KB")
    except ImportError:
        print(f"⚠ PNG generation skipped (kaleido not installed)")
        print("  Install with: pip install kaleido")
        print("  For now, you can screenshot the HTML file!")
    except Exception as e:
        print(f"⚠ PNG generation failed: {e}")
        print("  HTML version is still available!")
    
    return fig

def print_reddit_posting_info():
    """Print the Reddit posting information."""
    print("\n" + "="*70)
    print("REDDIT POSTING INFORMATION")
    print("="*70)
    
    print("\n📝 POST TITLE:")
    print("   [OC] GPU Transistor Count (1995-2025)")
    
    print("\n💬 REQUIRED COMMENT (Rule 3 - post this immediately after submitting):")
    print("-"*70)
    print("""Data Source: TechPowerUp GPU Database (via dbgpu Python library)
Tools: Python (pandas, plotly)
GitHub: https://github.com/YOUR_USERNAME/gpu-evolution-data

The chart shows the exponential growth of GPU transistor counts from 1995 to 2025, 
visualized with a linear scale to emphasize the dramatic increase in recent years. 
Data includes over 2,800 GPUs from manufacturers including NVIDIA, AMD, Intel, and others.""")
    print("-"*70)
    
    print("\n📋 CHECKLIST:")
    print("  ☐ Title is plain and descriptive (no sensationalism)")
    print("  ☐ Chart is uploaded as PNG image")
    print("  ☐ Tag post as [OC]")
    print("  ☐ Post the comment above immediately after submitting")
    print("  ☐ Include GitHub link in comment")
    print("  ☐ Best posting day: Avoid Mondays (personal data) and Thursdays (US politics)")
    
    print("\n🎯 TIPS FOR SUCCESS:")
    print("  • Linear scale makes growth look dramatic (good for Reddit!)")
    print("  • Clean title follows Rule 7 (no clickbait)")
    print("  • Comment cites data source (Rule 3)")
    print("  • Use the PNG image for best quality")
    print("  • Consider posting during peak Reddit hours (9am-2pm EST)")
    
    print("\n📂 OUTPUT FILES:")
    print(f"  • HTML (interactive): {OUTPUT_DIR}/transistor_count_reddit.html")
    print(f"  • PNG (for Reddit):   {OUTPUT_DIR}/transistor_count_reddit.png")
    
    print("\n" + "="*70 + "\n")

def main():
    """Generate Reddit-optimized transistor count visualization."""
    print("\n" + "="*70)
    print("GPU Transistor Count - Reddit Visualization Generator")
    print("="*70 + "\n")
    
    # Load data
    df = load_data()
    
    # Generate chart
    print("\nGenerating Reddit-optimized chart...")
    print("-" * 70)
    create_reddit_transistor_chart(df)
    
    # Print Reddit posting info
    print_reddit_posting_info()

if __name__ == "__main__":
    main()