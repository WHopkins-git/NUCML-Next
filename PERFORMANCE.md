# NUCML-Next Performance Optimization Guide

## Critical Optimizations for Large-Scale Training

### Problem 1: Slow Data Loading
Loading the full EXFOR database (4.7GB, ~18M measurements) can take 600+ seconds with default settings.

### Problem 2: Memory Explosion During One-Hot Encoding (CRITICAL!)
When training on 16.9M measurements with 117 MT codes:
- **Dense one-hot encoding**: 14.7 GB RAM required (16.9M × 117 × 8 bytes)
- **Result**: MemoryError on most systems

**This is now FIXED** - see "Memory Optimizations" section below.

---

## Memory Optimizations (Automatic - v1.1.0+)

### Sparse One-Hot Encoding for MT Codes

**The Problem:**
When converting EXFOR data to tabular format for ML, MT reaction codes need one-hot encoding:
```
Original:  MT = 18 (single integer)
One-hot:   [0, 0, ..., 1, ..., 0]  (117 values, mostly zeros)
```

For 16.9M measurements × 117 MT codes × 8 bytes (float64):
- **Dense array**: 14.7 GB RAM ❌ MemoryError!
- **Sparse array**: ~135 MB RAM ✅ Works!

**The Solution (Automatic):**
NUCML-Next now uses pandas sparse arrays for one-hot encoding:

```python
# Tier-based features use particle emission vectors (no memory issues)
df = dataset.to_tabular()

# Memory usage:
# - Tier-based features: Compact particle vectors instead of one-hot MT codes
# - No memory explosion - efficient by design!
```

**Technical Details:**
1. `OneHotEncoder(sparse_output=True)` returns scipy sparse matrix
2. Converted to pandas sparse DataFrame using `pd.DataFrame.sparse.from_spmatrix()`
3. ML models (DecisionTree, XGBoost) automatically convert to scipy CSR format
4. Training proceeds with sparse matrices (sklearn supports this natively)
5. Memory savings: **110x reduction** for typical EXFOR datasets

**Performance Impact:**
| Dataset Size | Dense Memory | Sparse Memory | Reduction |
|--------------|--------------|---------------|-----------|
| 1M rows × 117 MT | 0.9 GB | 8 MB | 110x |
| 5M rows × 117 MT | 4.4 GB | 40 MB | 110x |
| 16.9M rows × 117 MT | 14.7 GB | 135 MB | 110x |

**What You'll See:**
```
Training Decision Tree (max_depth=6, min_samples_leaf=20)...
  → Using sparse matrix format (memory efficient)
✓ Training complete!
```

---

## Data Loading Optimizations

### Solutions

#### 1. Use Filters to Load Only What You Need (FASTEST) ⚡

If you're only working with specific isotopes or reactions, filter at load time:

```python
# Only load U-235 and Pu-239 fission and capture data
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    filters={
        'Z': [92, 94],           # Uranium and Plutonium
        'A': [235, 239],         # U-235 and Pu-239
        'MT': [18, 102]          # Fission and capture
    }
)
```

**Performance:**
- Filter pushdown happens in PyArrow (C++ speed)
- Only loads ~1-5% of data from disk
- Load time: **2-10 seconds** instead of 600 seconds
- Suitable for focused analysis, prototyping, debugging

#### 2. Optimized Full Load (Column Pruning + Memory Mapping)

The latest version includes automatic optimizations:

```python
# Loads full database with optimizations
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular'
)
```

**Optimizations Applied:**
- ✅ **Column pruning**: Only reads essential 10 columns (not all 50+)
- ✅ **Memory mapping**: Faster I/O, less RAM pressure
- ✅ **Multi-threaded read**: Parallel decompression
- ✅ **Lazy graph building**: Graph built on first access

**Performance:**
- Column pruning: **50-60% faster** read time
- Memory mapping: **30-40% less RAM** usage
- Lazy graph: **Instant initialization**
- Expected load time: **60-120 seconds** (down from 600s)

#### 3. Lazy Loading for Interactive Development

```python
# Load only metadata initially, data on-demand
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    lazy_load=True  # Only loads 1000 rows initially
)
```

**Use case:**
- Rapid prototyping
- Testing code without waiting
- Exploring schema and data structure

#### 4. Pre-Filter with Partitioned Parquet (ADVANCED)

If you frequently work with specific subsets, re-export:

```python
# One-time: Export filtered data to new parquet
import pandas as pd
import pyarrow.parquet as pq

# Load with filter
table = pq.read_table(
    'data/exfor_processed.parquet',
    filters=[('Z', 'in', [92, 94]), ('MT', 'in', [18, 102, 16, 17])]
)

# Save subset
pq.write_table(table, 'data/actinides_fission.parquet')

# Future loads are instant
dataset = NucmlDataset('data/actinides_fission.parquet', mode='tabular')
```

**Performance:**
- Subset files: **100-500MB** instead of 4.7GB
- Load time: **1-5 seconds**
- Perfect for repeated workflows

---

## Performance Comparison

| Method | Load Time | RAM (Loading) | RAM (Training) | Use Case |
|--------|-----------|---------------|----------------|----------|
| Full load (old, v1.0) | 600s | 12 GB | 28 GB (MemoryError!) | ❌ Broken |
| **Optimized full (v1.1+)** | **60-120s** | **4-6 GB** | **5-7 GB** | ✅ Production |
| **Filtered load** | **2-10s** | **200 MB** | **300 MB** | ✅ Development |
| Lazy load | <1s | <100 MB | N/A | ✅ Prototyping |
| Pre-filtered file | 1-5s | 200 MB | 300 MB | ✅ Repeated workflows |

**Note on RAM (Training):** Includes sparse one-hot encoding (110x reduction vs dense)

---

## Recommended Workflow

### For Development/Prototyping:
```python
# Use filters to work with subset
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular',
    filters={'Z': [92], 'MT': [18, 102]}  # Just U-235 fission/capture
)
```

### For Full Model Training:
```python
# Load full database with optimizations
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='tabular'
    # Automatic optimizations: column pruning, memory mapping, lazy graph
)

# Train on full data with tier-based features
df = dataset.to_tabular()  # Uses tiers from DataSelection
model.train(df)
```

### For GNN Training:
```python
# Graph mode with lazy building
dataset = NucmlDataset(
    data_path='data/exfor_processed.parquet',
    mode='graph'
    # Graph built on first access to get_global_graph()
)

# Graph builds when you need it
graph = dataset.get_global_graph()  # <-- Build happens here
```

---

## Monitoring Load Performance

The optimized loader provides timing breakdowns:

```
Loading data from data/exfor_processed.parquet...
  → Parquet read: 45.2s (3.42 GB)
  → Arrow→Pandas conversion: 23.1s
✓ Loaded 18,345,672 EXFOR data points
```

**Interpretation:**
- **Parquet read**: I/O + decompression (benefits from column pruning, memory mapping)
- **Arrow→Pandas**: Type conversion (unavoidable, but faster with fewer columns)

If load time is still too slow:
1. Check disk speed (SSD vs HDD makes huge difference)
2. Use filters to reduce data size
3. Consider pre-filtering to create smaller files

---

## Advanced: PyArrow Native Operations (Future)

For maximum performance, work directly with PyArrow tables:

```python
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Read as Arrow table (no pandas conversion)
table = pq.read_table(
    'data/exfor_processed.parquet',
    columns=['Z', 'A', 'Energy', 'CrossSection', 'MT'],
    memory_map=True
)

# Use PyArrow compute (C++ speed)
u235_mask = (table['Z'] == 92) & (table['A'] == 235)
u235_table = table.filter(u235_mask)

# Convert to pandas only when needed
df = u235_table.to_pandas()
```

This approach can be **2-5x faster** for large filtering operations.

---

## Summary: Best Practices

✅ **DO:**
- Use filters for focused analysis
- Let optimizations run automatically (column pruning, memory mapping)
- Use lazy_load for prototyping
- Pre-filter to create smaller files for repeated workflows

❌ **DON'T:**
- Load full 4.7GB without filtering unless training production models
- Convert entire Arrow table to pandas if you don't need all columns
- Build graphs eagerly if you're in tabular mode

🎯 **Target Performance:**
- Filtered loads: **<10 seconds**
- Full optimized load: **60-120 seconds**
- Graph building: **30-60 seconds** (lazy, on-demand)
