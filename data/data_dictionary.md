# GPU Specifications Data Dictionary

This document defines all columns in the `gpu_specs.csv` dataset.

## Column Definitions

### Identification Fields

| Column | Type | Description | Example | Notes |
|--------|------|-------------|---------|-------|
| **Model** | String | GPU chip model designation | GP104 | Often the die/chip name, not the product name |
| **Manufacturer** | String | Company that designed the GPU | NVIDIA, AMD, Intel | Categorical |
| **Architecture** | String | GPU microarchitecture name | Pascal, RDNA 2, Xe | Used for grouping generations |
| **Foundry** | String | Semiconductor fab that manufactured the chip | TSMC, Samsung, GlobalFoundries | May be empty for older/unknown chips |

### Manufacturing Specifications

| Column | Type | Unit | Description | Example | Notes |
|--------|------|------|-------------|---------|-------|
| **ReleaseDate** | Date | YYYY-MM-DD | Official release/announcement date | 2016-05-27 | Use ISO 8601 format |
| **TransistorCount_B** | Float | Billions | Total number of transistors | 7.2 | Divide by 1,000,000,000 from spec sheets |
| **ProcessSize_nm** | Integer | Nanometers | Manufacturing process node | 16 | Marketing name, not physical gate length |
| **DieSize_mm2** | Float | mm² | Physical die area | 314 | Excludes package size |
| **TransistorDensity_M_mm2** | Float | Million/mm² | Transistors per square millimeter | 22.9 | Can be calculated: TransistorCount_B * 1000 / DieSize_mm2 |

### Performance Specifications

| Column | Type | Unit | Description | Example | Notes |
|--------|------|------|-------------|---------|-------|
| **TFLOPS** | Float | TFLOPS | Single-precision floating point performance | 8.9 | Theoretical peak (boost clock) |
| **MemoryBandwidth_GBs** | Float | GB/s | Memory subsystem bandwidth | 320.3 | Theoretical maximum |
| **BaseClock_MHz** | Integer | MHz | Base/reference clock speed | 1607 | Guaranteed minimum under load |
| **BoostClock_MHz** | Integer | MHz | Maximum boost clock speed | 1733 | May vary with thermals |
| **ShadingUnits** | Integer | Count | Number of shader processors/CUDA cores/stream processors | 2560 | Name varies by manufacturer |

### Power Specifications

| Column | Type | Unit | Description | Example | Notes |
|--------|------|------|-------------|---------|-------|
| **TDP_W** | Integer | Watts | Thermal Design Power | 180 | Official rated power consumption |

### Memory Specifications

| Column | Type | Unit | Description | Example | Notes |
|--------|------|------|-------------|---------|-------|
| **MemorySize_GB** | Float | GB | Video memory capacity | 8.0 | May vary by SKU |
| **MemoryType** | String | - | Memory technology type | GDDR5X, GDDR6, HBM2 | Categorical |

## Derived Metrics (Optional Calculations)

These can be computed from the base columns:

| Metric | Formula | Unit | Purpose |
|--------|---------|------|---------|
| **Performance per Watt** | TFLOPS / TDP_W | TFLOPS/W | Efficiency metric |
| **FLOPS per Transistor** | (TFLOPS * 1e12) / (TransistorCount_B * 1e9) | FLOPS/transistor | Computational efficiency |
| **Release Year** | Extract year from ReleaseDate | Year | For time-series grouping |

## Data Quality Standards

### Required Fields
Minimum required for an entry to be useful:
- Model
- Manufacturer
- ReleaseDate
- TransistorCount_B (or TDP_W or TFLOPS)

### Missing Data
- Use **empty cells** for unknown values
- Do NOT use: "Unknown", "N/A", "TBD", 0, -1
- Empty cells allow proper numerical analysis

### Data Validation Rules

1. **TransistorCount_B**: Should increase over time (Moore's Law)
2. **ProcessSize_nm**: Should generally decrease over time
3. **TDP_W**: Typically 15-600W for discrete GPUs
4. **TFLOPS**: Must be positive
5. **ReleaseDate**: Must be valid date, typically 1990-present
6. **TransistorDensity_M_mm2**: If provided, should match calculation

## Units Reference

| Abbreviation | Full Name | Conversion |
|--------------|-----------|------------|
| B | Billion | 1,000,000,000 |
| M | Million | 1,000,000 |
| nm | Nanometer | 10⁻⁹ meters |
| mm² | Square millimeter | - |
| MHz | Megahertz | 1,000,000 Hz |
| GHz | Gigahertz | 1,000 MHz |
| GB | Gigabyte | 1,024³ bytes (binary) or 10⁹ bytes (decimal) |
| GB/s | Gigabytes per second | - |
| W | Watt | - |
| TFLOPS | Teraflops | 10¹² floating point operations per second |

## Manufacturer Abbreviations

Common manufacturer values:
- **NVIDIA** (GeForce, Quadro, Tesla series)
- **AMD** (Radeon, formerly ATI)
- **Intel** (Arc, Xe, integrated graphics)
- **3dfx** (historical: Voodoo series)
- **Matrox** (historical)
- **S3 Graphics** (historical)

## Architecture Examples

### NVIDIA
- Fermi, Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell

### AMD
- TeraScale, Graphics Core Next (GCN), RDNA, RDNA 2, RDNA 3

### Intel
- Gen9, Gen11, Xe-LP, Xe-HPG (Arc)

## Data Sources

Recommended sources for verification:
1. **TechPowerUp GPU Database** - https://www.techpowerup.com/gpu-specs/
2. **WikiChip** - https://en.wikichip.org/
3. **Official manufacturer specifications**
4. **AnandTech** - https://www.anandtech.com/
5. **Tom's Hardware** - https://www.tomshardware.com/

## Version History

- **v1.0** (2026-01-06): Initial data dictionary

## Contributing Notes

When adding new GPU entries:
1. Cross-reference at least 2 sources
2. Prioritize official manufacturer specs
3. Note any discrepancies in commit messages
4. Use the exact chip/die name when possible
5. Include link to source in commit message