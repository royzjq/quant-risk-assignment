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

### 2. Data Preparation & Integration
Addresses the challenge of matching execution data with market data:

- **Time Series Alignment**: Uses `join_asof` to match execution records with nearest market snapshots, handling timestamp precision differences
- **Data Preprocessing**: Aggregates duplicate execution records based on business rules (averaging prices, summing quantities)
- **Coverage Analysis**: Provides detailed statistics on data completeness and matching success rates
- **Parquet Conversion**: Saves processed data in optimized format for subsequent analysis phases

### 3. Analysis Modules

#### Market Impact Analysis
**Logic**: Checks if executions violate market depth or price boundaries using specific rules:
- **Buy Side (sde=1)**: `eqty > askVol-1` or `vwap > askPrice-1` indicates potential market impact
- **Sell Side (sde=2)**: `eqty > bidVol-1` or `vwap < bidPrice-1` indicates potential market impact
- **1s Price Impact**: Measures actual price movement 1 second after execution, only flags impacts > 1.5σ
- **Timestamp Export**: Saves violation timestamps to JSON for further investigation

#### Slippage Analysis
**Logic**: Examines if slippage correlates with market conditions
- **Slippage Calculation**: Buy: `slippage = vwap - mid_price`, Sell: `slippage = mid_price - vwap` (in basis points)
- **Order Size Correlation**: Plots slippage vs execution quantity to see if larger orders have worse slippage
- **Volatility Correlation**: Checks if slippage increases during high volatility periods
- **Liquidity Impact**: Analyzes if low liquidity (bid/ask volumes) leads to higher slippage

#### Latency Analysis
**Logic**: Analyzes time delays in order processing and execution
- **Multi-Stage Timing**: Tracks latency across different execution phases
- **Outlier Detection**: Identifies abnormal latency patterns using statistical thresholds
- **Distribution Analysis**: Provides percentile-based latency metrics

#### Fee Analysis
**Logic**: Validates fee structure and identifies outliers
- **Fee Validation**: Checks if fees are negative (costs) as expected
- **Fee Rate Analysis**: Calculates fee as percentage of execution value to identify expensive trades
- **Outlier Detection**: Flags unusually high or positive fees that may indicate data errors

#### Spread Analysis
**Logic**: Checks if executions occur during wide spread periods
- **Spread Timeline Plot**: Shows bid/ask spreads over time with execution points marked at y=0
- **Wide Spread Detection**: Identifies if trades happen when spreads are unusually wide (bad timing)
- **Distribution Verification**: Creates histograms to double-check spread patterns and execution frequency

## Visualization

Uses **Plotly** for interactive charts that can be easily integrated into production dashboards:
- **HTML Output**: Interactive charts for detailed analysis
- **PNG Export**: Static images for reports
- **Downsampling**: Configurable intervals for large datasets to maintain performance

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

### Quick Start
1. Put `exec.csv` and market data CSV files in the `data/` folder
2. Run the pipeline:
```bash
python main.py
```

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