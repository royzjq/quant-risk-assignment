# Quantitative Risk Analysis Framework

A high-performance modular quantitative risk analysis framework for execution and market data analysis, designed for speed and scalability.

## Project Architecture & Design Philosophy

### Performance-First Design
This project was specifically designed to address performance limitations of traditional pandas-based analysis:

- **Polars-First Approach**: Completely replaced pandas with Polars for all data operations, providing 10-50x speed improvements through vectorized operations and lazy evaluation
- **Single Data Loading**: Uses a shared processed dataset across all analysis modules to eliminate redundant I/O operations
- **Parquet Optimization**: Converts CSV data to Parquet format for columnar storage, enabling faster queries and reduced memory footprint
- **Vectorized Calculations**: All statistical calculations leverage Polars' native vectorized operations for maximum performance

### Modular Framework Design
- **Independent Analysis Modules**: Each analysis component (`slippage_analyzer`, `latency_analyzer`, `fee_analyzer`, `market_impact_analyzer`, `spread_analyzer`) operates independently
- **Easy Extension**: Adding new analysis modules requires minimal integration effort
- **Configurable Pipeline**: Components can be enabled/disabled based on analysis requirements
- **Standardized Interface**: All modules follow consistent input/output patterns

## Project Workflow

### 1. Data Validation Phase
The framework performs comprehensive data quality checks before analysis:

- **Missing Values Detection**: Identifies null/empty values across all columns with detailed location tracking
- **Data Type Validation**: Ensures numerical columns contain valid numeric data, converts with error handling
- **Value Range Checks**: Validates that prices, quantities, and fees fall within expected ranges
- **Business Logic Validation**: Checks consistency between order quantities, executed quantities, and remaining quantities
- **Event Type Verification**: Ensures execution events match expected trading event types
- **Exchange Validation**: Confirms all records originate from expected exchanges

### 2. Data Preparation & Integration
Addresses the challenge of matching execution data with market data:

- **Time Series Alignment**: Uses `join_asof` to match execution records with nearest market snapshots, handling timestamp precision differences
- **Data Preprocessing**: Aggregates duplicate execution records based on business rules (averaging prices, summing quantities)
- **Coverage Analysis**: Provides detailed statistics on data completeness and matching success rates
- **Parquet Conversion**: Saves processed data in optimized format for subsequent analysis phases

### 3. Analysis Modules

#### Market Impact Analysis
**Logic**: Detects potential market manipulation or significant market impact violations
- **Quantity Violations**: Identifies executions exceeding available market depth (bid/ask volumes)
- **Price Violations**: Detects VWAP prices that breach bid/ask price boundaries
- **Statistical Significance**: Only flags impacts exceeding 1.5 sigma thresholds to reduce false positives
- **Side-Specific Logic**: Applies different validation rules for buy vs sell orders

#### Slippage Analysis
**Logic**: Measures execution efficiency relative to mid-market prices
- **Side-Aware Calculation**: Buy orders: `slippage = vwap - mid_price`, Sell orders: `slippage = mid_price - vwap`
- **Basis Points Conversion**: Normalizes slippage as percentage of mid price for comparability
- **Correlation Analysis**: Examines relationships between slippage and order size, market volatility, liquidity

#### Latency Analysis
**Logic**: Analyzes time delays in order processing and execution
- **Multi-Stage Timing**: Tracks latency across different execution phases
- **Outlier Detection**: Identifies abnormal latency patterns using statistical thresholds
- **Distribution Analysis**: Provides percentile-based latency metrics

#### Fee Analysis
**Logic**: Comprehensive trading cost analysis
- **Fee Structure Validation**: Ensures fees follow expected patterns (typically negative values representing costs)
- **Cost Efficiency Metrics**: Calculates fee rates relative to execution values
- **Outlier Detection**: Identifies unusual fee patterns that may indicate errors

#### Spread Analysis
**Logic**: Market microstructure analysis focusing on bid-ask dynamics
- **Real-Time Spread Tracking**: Monitors bid-ask spread evolution over time
- **Execution Impact Visualization**: Shows execution points overlaid on spread timelines
- **Distribution Analysis**: Analyzes spread distribution patterns and identifies anomalies

## Visualization Strategy

### Performance-Optimized Plotting
- **Selective Downsampling**: Uses configurable downsampling intervals for time series plots to maintain visual clarity while reducing rendering time
- **Interactive HTML Output**: Generates Plotly HTML files for detailed interactive analysis
- **Static PNG Export**: Creates high-resolution PNG images for reports and presentations

### Required Visualizations
1. **Spread Timeline with Executions**: Time series plot showing bid/ask spreads with execution points marked at y=0
2. **Slippage Distribution**: Histogram of slippage values in basis points with statistical annotations
3. **Correlation Plots**: Scatter plots showing relationships between slippage and market factors (order size, volatility, liquidity)
4. **Distribution Analysis**: Multi-panel plots showing spread distributions with execution frequency overlays

### Design Requirements
- **Consistent Styling**: All plots use `plotly_white` template for professional appearance
- **Hover Information**: Detailed hover tooltips showing execution details and market conditions
- **Range Sliders**: Time series plots include navigation controls for zooming
- **Statistical Annotations**: Plots include mean, median, standard deviation, and correlation statistics

## Technical Specifications

### Dependencies
- **Polars (≥0.19.0)**: Primary data processing engine
- **Plotly (≥5.15.0)**: Visualization framework
- **NumPy (≥1.24.0)**: Numerical computations
- **SciPy (≥1.10.0)**: Statistical analysis
- **Scikit-learn (≥1.3.0)**: Machine learning algorithms for correlation analysis

### File Structure
```
quant-risk-assignment/
├── main.py                          # Pipeline orchestrator
├── validation/
│   ├── __init__.py
│   └── exec_validator.py           # Data quality validation
├── data_prep/
│   ├── __init__.py
│   └── data_processor.py           # Data loading, cleaning, merging
├── analysis/
│   ├── __init__.py
│   ├── market_impact_analyzer.py   # Market impact detection
│   ├── slippage_analyzer.py        # Execution efficiency analysis
│   ├── latency_analyzer.py         # Timing analysis
│   ├── fee_analyzer.py             # Cost analysis
│   └── spread_analyzer.py          # Market microstructure analysis
├── results/                        # Generated outputs
│   ├── images/                     # Visualization files
│   ├── *_processed_data.parquet    # Optimized datasets
│   └── *.json                      # Analysis results
└── data/                           # Market data directory
```

## Usage

### Complete Pipeline
```bash
# Run full analysis pipeline
python main.py

# Process specific symbols
python main.py --symbols BTCUSDT,ETHUSDT

# Force data reprocessing
python main.py --force-reprocess

# Disable specific analysis modules
python main.py --disable-spread-analysis
```

### Module-Level Usage
```python
# Independent module usage
from analysis.slippage_analyzer import run_comprehensive_slippage_analysis_with_data
from data_prep.data_processor import prepare_symbol_data

# Load and process data
df, status = prepare_symbol_data('exec.csv', 'data', 'BTCUSDT')

# Run specific analysis with shared data
results = run_comprehensive_slippage_analysis_with_data(
    df=df, 
    output_dir="results/images",
    symbol="BTCUSDT"
)
```

## Performance Characteristics

- **Data Loading**: 10-50x faster than pandas for large datasets
- **Memory Efficiency**: Lazy evaluation reduces memory consumption by 30-60%
- **Analysis Speed**: Vectorized operations complete analysis in seconds rather than minutes
- **Scalability**: Framework handles datasets from MBs to GBs without architecture changes

## Output Files

- **Validation Results**: `exec_validation.json` - Comprehensive data quality report
- **Processed Data**: `{symbol}_processed_data.parquet` - Optimized merged datasets
- **Analysis Results**: Various JSON files containing violation timestamps and metrics
- **Visualizations**: HTML and PNG files in `results/images/` directory 