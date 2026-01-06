# GPU Evolution Data

A comprehensive historical database tracking the evolution of graphics processing units (GPUs) from early designs to modern architectures. This repository contains detailed specifications including transistor counts, power consumption, performance metrics, and manufacturing details for analyzing Moore's Law and GPU efficiency trends.

## 📊 Dataset Overview

The database includes key metrics for GPUs across multiple generations:

- **Identification**: Model name, manufacturer, architecture, foundry
- **Manufacturing**: Release date, process size, die size, transistor count & density
- **Performance**: TFLOPS, memory bandwidth, clock speeds, shading units
- **Power**: TDP (Thermal Design Power)
- **Memory**: Size, type (GDDR5, GDDR6, HBM, etc.)
- **Derived Metrics**: Performance per watt, FLOPS per transistor

## 🗂️ Repository Structure

```
gpu-evolution-data/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
├── data/
│   ├── gpu_specs.csv        # Main dataset
│   └── data_dictionary.md   # Column definitions and metadata
├── scripts/
│   ├── requirements.txt     # Python dependencies
│   ├── collect_data.py      # Data collection from dbgpu
│   └── visualize.py         # Visualization script
└── output/                  # Generated charts and analysis
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpu-evolution-data.git
cd gpu-evolution-data
```

2. Install dependencies:
```bash
pip install -r scripts/requirements.txt
```

### Collect GPU Data

Use the `collect_data.py` script to populate the CSV with GPU specifications from the dbgpu database:

```bash
# Collect all NVIDIA GPUs from 2015 onwards
python scripts/collect_data.py --manufacturer NVIDIA --start-year 2015

# Collect both NVIDIA and AMD GPUs
python scripts/collect_data.py --manufacturer NVIDIA AMD

# Collect a random sample of 100 GPUs for testing
python scripts/collect_data.py --sample 100

# Collect specific architectures
python scripts/collect_data.py --manufacturer NVIDIA --architecture Pascal Ampere Turing
```

### Generate Visualizations

Run the visualization script to generate charts:

```bash
python scripts/visualize.py
```

Charts will be saved to the `output/` directory.

## 📈 Available Visualizations

The visualization script generates several charts:

1. **Transistor Count Over Time** - Moore's Law progression
2. **TDP (Power) Over Time** - Power consumption trends
3. **Performance (TFLOPS) Over Time** - Computational power growth
4. **Performance per Watt** - Efficiency improvements
5. **Process Size Evolution** - Manufacturing node progression
6. **Die Size Trends** - Chip size over time
7. **Memory Bandwidth Growth** - Data throughput evolution
8. **Architecture Comparison** - Performance across different architectures

## 🔍 Data Sources

GPU specifications are collected from:
- TechPowerUp GPU Database
- Manufacturer specifications (NVIDIA, AMD, Intel)
- AnandTech reviews and analyses
- WikiChip semiconductor database

## 📝 Contributing

Contributions are welcome! To add GPU data:

1. Fork the repository
2. Add entries to `data/gpu_specs.csv` following the format in `data/data_dictionary.md`
3. Ensure data accuracy with official sources
4. Submit a pull request

### Data Quality Guidelines

- Include source/reference for specifications
- Use consistent units (see data dictionary)
- Mark unknown values as empty (not "Unknown" or "N/A")
- Verify release dates from official announcements

## 📊 Data Format

The main dataset (`gpu_specs.csv`) uses the following format:

```csv
Model,Manufacturer,Architecture,Foundry,ReleaseDate,TransistorCount_B,TDP_W,ProcessSize_nm,DieSize_mm2,TFLOPS,MemoryBandwidth_GBs,BaseClock_MHz,BoostClock_MHz,TransistorDensity_M_mm2,MemorySize_GB,MemoryType,ShadingUnits
```

See `data/data_dictionary.md` for detailed column definitions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The dataset compilation is released under CC0 (public domain) as it contains factual specifications.

## 🙏 Acknowledgments

- TechPowerUp for maintaining comprehensive GPU databases
- Hardware review sites for detailed specifications
- The semiconductor industry for public technical documentation

## 📬 Contact

For questions, suggestions, or corrections, please open an issue on GitHub.

---

**Note**: This is a community-driven project. While we strive for accuracy, specifications should be verified against official sources for critical applications.