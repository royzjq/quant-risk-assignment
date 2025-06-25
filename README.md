# Quantitative Risk Analysis Project
# 量化风险分析项目

A modular quantitative risk analysis framework for execution and market data analysis.

## Project Structure / 项目结构

```
quant_risk_assignment/
├── main.py                 # Main pipeline orchestrator / 主程序
├── validation/             # Data validation modules / 数据验证模块
│   ├── __init__.py
│   └── exec_validator.py   # Execution data validator / 执行数据验证器
├── data_prep/             # Data preparation modules / 数据准备模块
│   ├── __init__.py
│   └── data_processor.py  # Data processing and merging / 数据处理和合并
├── analysis/              # Analysis modules / 分析模块
│   ├── __init__.py
│   └── market_impact_analyzer.py  # Market impact analysis / 市场影响分析
├── results/               # Output directory / 输出目录
│   ├── exec_validation.json       # Validation results / 验证结果
│   ├── btcusdt_processed_data.parquet  # Processed data / 处理后数据
│   └── violation_timestamps_*.json     # Violation timestamps / 违规时间戳
├── data/                  # Market data directory / 市场数据目录
└── exec.csv              # Execution data / 执行数据
```

## Features / 功能特性

1. **Data Validation / 数据验证**: Comprehensive validation of execution data
2. **Data Preparation / 数据准备**: Automated data loading, cleaning, and merging
3. **Market Impact Analysis / 市场影响分析**: Detection of potential market impact violations
4. **Modular Design / 模块化设计**: Each component can be used independently
5. **Performance Optimization / 性能优化**: Uses Parquet format for fast data loading

## Usage / 使用方法

### Basic Usage / 基础用法

```bash
# Run the complete pipeline / 运行完整流水线
python main.py

# Run with custom files / 使用自定义文件运行
python main.py --exec-file exec.csv --market-data-dir data

# Force reprocessing of data / 强制重新处理数据
python main.py --force-reprocess
```

### Module Usage / 模块使用

#### 1. Data Validation / 数据验证
```python
from validation.exec_validator import validate_exec_csv
results = validate_exec_csv('exec.csv')
```

#### 2. Data Preparation / 数据准备
```python
from data_prep.data_processor import prepare_btcusdt_data, save_to_parquet
merged_df, status = prepare_btcusdt_data('exec.csv', 'data')
save_to_parquet(merged_df, 'results/processed_data.parquet')
```

#### 3. Market Impact Analysis / 市场影响分析
```python
from analysis.market_impact_analyzer import analyze_market_impact_violations
results = analyze_market_impact_violations(merged_df, save_timestamps=True)
```

## Output Files / 输出文件

- **exec_validation.json**: Detailed validation results with anomalies
- **btcusdt_processed_data.parquet**: Processed and merged market-execution data
- **violation_timestamps_*.json**: Timestamps of market impact violations

## Installation / 安装

### Prerequisites / 前置要求
- Python 3.8 or higher / Python 3.8 或更高版本

### Install Dependencies / 安装依赖
```bash
# Install from requirements.txt / 从requirements.txt安装
pip install -r requirements.txt

# Or install packages individually / 或单独安装包
pip install polars>=0.19.0 pandas>=2.0.0 numpy>=1.24.0 plotly>=5.15.0 scipy>=1.10.0 scikit-learn>=1.3.0
```

## Dependencies / 依赖

### Core Dependencies / 核心依赖
- **polars**: Fast DataFrame library for data processing / 快速数据处理库
- **pandas**: Data analysis and manipulation / 数据分析和操作
- **numpy**: Numerical computing / 数值计算
- **plotly**: Interactive visualizations / 交互式可视化
- **scipy**: Scientific computing / 科学计算
- **scikit-learn**: Machine learning algorithms / 机器学习算法

### Optional Dependencies / 可选依赖
- **kaleido**: Static image export for Plotly / Plotly静态图像导出
- **ipython**: Enhanced interactive Python / 增强交互式Python
- **jupyter**: Notebook environment / 笔记本环境

## Analysis Results / 分析结果

The pipeline identifies potential market impact violations in four categories:
流水线识别四类潜在市场影响违规：

1. **eqty violation (sde=2, eqty > bidVol-1)**: Sell execution quantity exceeds bid volume
2. **eqty violation (sde=1, eqty > askVol-1)**: Buy execution quantity exceeds ask volume  
3. **vwap violation (sde=2, vwap < bidPrice-1)**: Sell VWAP below bid price
4. **vwap violation (sde=1, vwap > askPrice-1)**: Buy VWAP above ask price

Each violation includes timestamps for further analysis.

## Performance / 性能

- Uses Polars for fast data processing
- Parquet format for efficient storage and loading
- Modular design allows selective execution of analysis steps
- Caching of processed data to avoid recomputation 