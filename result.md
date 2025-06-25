============================================================
QUANTITATIVE RISK ANALYSIS PIPELINE
============================================================
Processing symbols: BTCUSDT, DOTUSDT, EOSUSDT, ETHUSDT, XRPUSDT

========================================
STEP 1: DATA VALIDATION
Successfully loaded data\exec.csv. Total rows: 13995
Checking for missing values...
  - No missing values found in any column.

Checking data types and converting...
  - All numerical columns seem to have correct data types.

Performing value range and consistency checks...
  - EventType: All values are in the expected list.
  - ech: All values are 'Binance'.
  - sde: All values are 1 or 2 (or NaN).
  - Column 'px': All values are non-negative.
  - Column 'qty': All values are non-negative.
  - Column 'epx': All values are non-negative.
  - Column 'eqty': All values are non-negative.
  - Column 'vwap': All values are non-negative.
  - Column 'rqty': All values are non-negative.
  - Column 'fee': 2 non-negative values found (expected strictly negative).
  - Quantity Consistency: 10 inconsistent rows found.

Validation complete.
Validation report saved to results/exec_validation.json

============================================================
PROCESSING BTCUSDT
============================================================

========================================
STEP 2: DATA PREPARATION FOR BTCUSDT
Processed data found at results/btcusdt_processed_data.parquet
Loading existing processed data for BTCUSDT analysis...
Loading processed data from results/btcusdt_processed_data.parquet...
Loaded 2459489 records in 0.47 seconds

========================================
STEP 3: MARKET IMPACT ANALYSIS FOR BTCUSDT

=== Market Impact Analysis ===

=== Market Impact Data Validation ===
eqty violations (sde=2, eqty > bidVol-1): 4
eqty violations (sde=1, eqty > askVol-1): 0
vwap violations (sde=2, vwap < bidPrice-1): 0
vwap violations (sde=1, vwap > askPrice-1): 0
Market Impact Data Validation Complete.

========================================
STEP 4: SLIPPAGE ANALYSIS FOR BTCUSDT
============================================================
COMPREHENSIVE SLIPPAGE ANALYSIS
============================================================
Using processed data with 2459489 records.
Calculating slippage...
Calculated slippage for 1081 execution records
Calculating market volatility with 100-period window...
Calculating liquidity measure with 10-period average...
Generating slippage summary statistics...

Generating visualizations...
Creating slippage distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage distribution plot saved to results/images/
Creating slippage vs order size plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs order size plot saved to results/images/
Creating slippage vs volatility plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs volatility plot saved to results/images/
Creating slippage vs liquidity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs liquidity plot saved to results/images/

============================================================
SLIPPAGE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.76 seconds
Total execution records analyzed: 1081
Mean slippage: -0.21 bps
Median slippage: -0.02 bps
Standard deviation: 0.39 bps
Slippage vs Order Size correlation: -0.061
Slippage vs Volatility correlation: -0.242
Slippage vs Liquidity correlation: 0.041
Visualizations saved to: results/images/
============================================================

========================================
STEP 5: LATENCY ANALYSIS FOR BTCUSDT
============================================================
COMPREHENSIVE LATENCY ANALYSIS
============================================================
Using processed data with 2459489 records.
Calculating latencies...
Calculated latencies for 1081 execution records
Generating latency summary statistics...

Generating visualizations...
Creating latency distribution plots...
Warning: Could not save PNG image. HTML version saved successfully.
Latency distribution plots saved to results/images/

Detecting latency outliers...
Found 31 exchange_processing_latency outliers (threshold: 2.5σ)
exchange_processing_latency outliers saved to results/btcusdt_exchange_processing_latency_outliers_1750867993.json
Found 35 network_latency outliers (threshold: 2.5σ)
network_latency outliers saved to results/btcusdt_network_latency_outliers_1750867993.json
Found 19 strategy_processing_latency outliers (threshold: 2.5σ)
strategy_processing_latency outliers saved to results/btcusdt_strategy_processing_latency_outliers_1750867993.json
============================================================

============================================================
LATENCY ANALYSIS SUMMARY
============================================================
Analysis completed in 0.31 seconds

Exchange Processing Latency:
  - Count: 1081
  - Mean: 24330.64 μs
  - Median: 33531.00 μs
  - P95: 38050.00 μs
  - P99: 89702.40 μs
  - Outliers (>2.5σ): 31 (2.87%)

Network Latency:
  - Count: 1081
  - Mean: 94.58 μs
  - Median: 75.00 μs
  - P95: 210.00 μs
  - P99: 316.80 μs
  - Outliers (>2.5σ): 35 (3.24%)

Strategy Processing Latency:
  - Count: 1081
  - Mean: 16030.53 μs
  - Median: 15000.00 μs
  - P95: 33000.00 μs
  - P99: 53000.00 μs
  - Outliers (>2.5σ): 19 (1.76%)

Visualization saved to: results/images/btcusdt_latency_distributions.html
Outlier files saved with prefix: btcusdt_
============================================================

========================================
STEP 6: FEE ANALYSIS FOR BTCUSDT
============================================================
COMPREHENSIVE FEE ANALYSIS
============================================================
Using processed data with 2459489 records.
Preparing fee analysis data...
Prepared fee analysis data for 1081 execution records
Detecting fee outliers with 2.0 sigma threshold...
Found 12 fee outliers out of 1081 records (1.11%)
Generating fee summary statistics...

Generating visualizations...
Creating fee distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee distribution plot saved to results/images/
Creating fee vs quantity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs quantity plot saved to results/images/
Creating fee vs VWAP plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs VWAP plot saved to results/images/
Creating fee vs notional value plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs notional value plot saved to results/images/
Analyzing fee outliers...
Fee outlier analysis exported to results/btcusdt_fee_outliers_20250626_001313.json

============================================================
FEE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.30 seconds
Total execution records analyzed: 1081
Fee statistics:
  - Mean: -0.009184
  - Median: -0.010566
  - Std: 0.004962
Correlations:
  - Fee vs Quantity: -1.000
  - Fee vs VWAP: -0.007
  - Fee vs Notional: -1.000
Outlier analysis:
  - Outliers found: 12 (1.1%)
  - Mean outlier fee: -0.026999
  - Mean normal fee: -0.008984
Visualizations saved to: results/images/
============================================================

========================================
STEP 7: SPREAD ANALYSIS FOR BTCUSDT
============================================================
COMPREHENSIVE SPREAD ANALYSIS
============================================================
Using processed data with 2459489 records.
Calculating bid and ask spreads...
Spread calculations completed
Preparing data for plotting...
Market data records: 2459489
Execution records: 1081
Generating summary statistics...

Generating visualizations...
Creating spread visualization...
Downsampling market data for visualization (interval: 100)...
Original market data points: 2459489, Sampled points: 24595
Warning: Could not save PNG image. HTML version saved successfully.
Visualization saved to results\images\btcusdt_spread_execution_visualization.png
Interactive HTML saved to results\images\btcusdt_spread_execution_visualization.html
Creating bid-ask spread distribution chart...
Market data: 2458408 records, 23940 (1.0%) exceed 99% percentile
Execution data: 1081 records, 13 (1.2%) exceed 99% percentile
99% percentile value: 0.900000
Warning: Could not save PNG image. HTML version saved successfully.
Bid-ask spread distribution saved to results\images\btcusdt_bidask_spread_distribution.png
Interactive HTML saved to results\images\btcusdt_bidask_spread_distribution.html
Exporting trades with bid-ask spread > 0.2 USDT...
Exported 68 trades with large spreads to results\btcusdt_large_spread_trades_0.2usdt_20250626_001314.json

============================================================
SPREAD ANALYSIS SUMMARY
============================================================
Analysis completed in 0.59 seconds
Market data records: 2459489
Execution records: 1081
Bid-ask spread (mean/median): 0.129286 / 0.100000 USDT
Buy orders: 500, Sell orders: 581
Visualizations saved to: results/images/
============================================================

============================================================
BTCUSDT PIPELINE EXECUTION SUMMARY
============================================================
Execution time: 2.50 seconds
Data preparation: LOADED_FOR_ANALYSIS
Market impact analysis: COMPLETE
Slippage analysis: COMPLETE
Latency analysis: COMPLETE
Fee analysis: COMPLETE
Spread analysis: COMPLETE

============================================================
PROCESSING DOTUSDT
============================================================

========================================
STEP 2: DATA PREPARATION FOR DOTUSDT
Processed data found at results/dotusdt_processed_data.parquet
Loading existing processed data for DOTUSDT analysis...
Loading processed data from results/dotusdt_processed_data.parquet...
Loaded 1877854 records in 0.26 seconds

========================================
STEP 3: MARKET IMPACT ANALYSIS FOR DOTUSDT

=== Market Impact Analysis ===

=== Market Impact Data Validation ===
eqty violations (sde=2, eqty > bidVol-1): 14
eqty violations (sde=1, eqty > askVol-1): 36
vwap violations (sde=2, vwap < bidPrice-1): 0
vwap violations (sde=1, vwap > askPrice-1): 1
Market Impact Data Validation Complete.

========================================
STEP 4: SLIPPAGE ANALYSIS FOR DOTUSDT
============================================================
COMPREHENSIVE SLIPPAGE ANALYSIS
============================================================
Using processed data with 1877854 records.
Calculating slippage...
Calculated slippage for 3045 execution records
Calculating market volatility with 100-period window...
Calculating liquidity measure with 10-period average...
Generating slippage summary statistics...

Generating visualizations...
Creating slippage distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage distribution plot saved to results/images/
Creating slippage vs order size plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs order size plot saved to results/images/
Creating slippage vs volatility plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs volatility plot saved to results/images/
Creating slippage vs liquidity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs liquidity plot saved to results/images/

============================================================
SLIPPAGE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.60 seconds
Total execution records analyzed: 3045
Mean slippage: -1.04 bps
Median slippage: -0.71 bps
Standard deviation: 0.85 bps
Slippage vs Order Size correlation: 0.043
Slippage vs Volatility correlation: -0.070
Slippage vs Liquidity correlation: 0.027
Visualizations saved to: results/images/
============================================================

========================================
STEP 5: LATENCY ANALYSIS FOR DOTUSDT
============================================================
COMPREHENSIVE LATENCY ANALYSIS
============================================================
Using processed data with 1877854 records.
Calculating latencies...
Calculated latencies for 3045 execution records
Generating latency summary statistics...

Generating visualizations...
Creating latency distribution plots...
Warning: Could not save PNG image. HTML version saved successfully.
Latency distribution plots saved to results/images/

Detecting latency outliers...
Found 91 exchange_processing_latency outliers (threshold: 2.5σ)
exchange_processing_latency outliers saved to results/dotusdt_exchange_processing_latency_outliers_1750867995.json
Found 15 network_latency outliers (threshold: 2.5σ)
network_latency outliers saved to results/dotusdt_network_latency_outliers_1750867995.json
Found 64 strategy_processing_latency outliers (threshold: 2.5σ)
strategy_processing_latency outliers saved to results/dotusdt_strategy_processing_latency_outliers_1750867995.json
============================================================

============================================================
LATENCY ANALYSIS SUMMARY
============================================================
Analysis completed in 0.17 seconds

Exchange Processing Latency:
  - Count: 3045
  - Mean: 17626.01 μs
  - Median: 3193.00 μs
  - P95: 64548.20 μs
  - P99: 170485.80 μs
  - Outliers (>2.5σ): 91 (2.99%)

Network Latency:
  - Count: 3045
  - Mean: 27.97 μs
  - Median: 24.00 μs
  - P95: 52.00 μs
  - P99: 77.00 μs
  - Outliers (>2.5σ): 15 (0.49%)

Strategy Processing Latency:
  - Count: 3045
  - Mean: 24384.89 μs
  - Median: 19000.00 μs
  - P95: 65000.00 μs
  - P99: 126120.00 μs
  - Outliers (>2.5σ): 64 (2.10%)

Visualization saved to: results/images/dotusdt_latency_distributions.html
Outlier files saved with prefix: dotusdt_
============================================================

========================================
STEP 6: FEE ANALYSIS FOR DOTUSDT
============================================================
COMPREHENSIVE FEE ANALYSIS
============================================================
Using processed data with 1877854 records.
Preparing fee analysis data...
Prepared fee analysis data for 3045 execution records
Detecting fee outliers with 2.0 sigma threshold...
Found 135 fee outliers out of 3045 records (4.43%)
Generating fee summary statistics...

Generating visualizations...
Creating fee distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee distribution plot saved to results/images/
Creating fee vs quantity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs quantity plot saved to results/images/
Creating fee vs VWAP plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs VWAP plot saved to results/images/
Creating fee vs notional value plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs notional value plot saved to results/images/
Analyzing fee outliers...
Fee outlier analysis exported to results/dotusdt_fee_outliers_20250626_001315.json

============================================================
FEE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.29 seconds
Total execution records analyzed: 3045
Fee statistics:
  - Mean: -0.002915
  - Median: -0.002462
  - Std: 0.001750
Correlations:
  - Fee vs Quantity: -1.000
  - Fee vs VWAP: -0.129
  - Fee vs Notional: -1.000
Outlier analysis:
  - Outliers found: 135 (4.4%)
  - Mean outlier fee: -0.008559
  - Mean normal fee: -0.002653
Visualizations saved to: results/images/
============================================================

========================================
STEP 7: SPREAD ANALYSIS FOR DOTUSDT
============================================================
COMPREHENSIVE SPREAD ANALYSIS
============================================================
Using processed data with 1877854 records.
Calculating bid and ask spreads...
Spread calculations completed
Preparing data for plotting...
Market data records: 1877854
Execution records: 3045
Generating summary statistics...

Generating visualizations...
Creating spread visualization...
Downsampling market data for visualization (interval: 100)...
Original market data points: 1877854, Sampled points: 18779
Warning: Could not save PNG image. HTML version saved successfully.
Visualization saved to results\images\dotusdt_spread_execution_visualization.png
Interactive HTML saved to results\images\dotusdt_spread_execution_visualization.html
Creating bid-ask spread distribution chart...
Market data: 1874809 records, 11835 (0.6%) exceed 99% percentile
Execution data: 3045 records, 47 (1.5%) exceed 99% percentile
99% percentile value: 0.002000
Warning: Could not save PNG image. HTML version saved successfully.
Bid-ask spread distribution saved to results\images\dotusdt_bidask_spread_distribution.png
Interactive HTML saved to results\images\dotusdt_bidask_spread_distribution.html
Exporting trades with bid-ask spread > 0.2 USDT...
No trades found with bid-ask spread > 0.2 USDT

============================================================
SPREAD ANALYSIS SUMMARY
============================================================
Analysis completed in 0.49 seconds
Market data records: 1877854
Execution records: 3045
Bid-ask spread (mean/median): 0.001041 / 0.001000 USDT
Buy orders: 1881, Sell orders: 1164
Visualizations saved to: results/images/
============================================================

============================================================
DOTUSDT PIPELINE EXECUTION SUMMARY
============================================================
Execution time: 1.87 seconds
Data preparation: LOADED_FOR_ANALYSIS
Market impact analysis: COMPLETE
Slippage analysis: COMPLETE
Latency analysis: COMPLETE
Fee analysis: COMPLETE
Spread analysis: COMPLETE

============================================================
PROCESSING EOSUSDT
============================================================

========================================
STEP 2: DATA PREPARATION FOR EOSUSDT
Processed data found at results/eosusdt_processed_data.parquet
Loading existing processed data for EOSUSDT analysis...
Loading processed data from results/eosusdt_processed_data.parquet...
Loaded 1516224 records in 0.17 seconds

========================================
STEP 3: MARKET IMPACT ANALYSIS FOR EOSUSDT

=== Market Impact Analysis ===

=== Market Impact Data Validation ===
eqty violations (sde=2, eqty > bidVol-1): 0
eqty violations (sde=1, eqty > askVol-1): 1
vwap violations (sde=2, vwap < bidPrice-1): 0
vwap violations (sde=1, vwap > askPrice-1): 2
Market Impact Data Validation Complete.

========================================
STEP 4: SLIPPAGE ANALYSIS FOR EOSUSDT
============================================================
COMPREHENSIVE SLIPPAGE ANALYSIS
============================================================
Using processed data with 1516224 records.
Calculating slippage...
Calculated slippage for 1514 execution records
Calculating market volatility with 100-period window...
Calculating liquidity measure with 10-period average...
Generating slippage summary statistics...

Generating visualizations...
Creating slippage distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage distribution plot saved to results/images/
Creating slippage vs order size plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs order size plot saved to results/images/
Creating slippage vs volatility plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs volatility plot saved to results/images/
Creating slippage vs liquidity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs liquidity plot saved to results/images/

============================================================
SLIPPAGE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.54 seconds
Total execution records analyzed: 1514
Mean slippage: -4.55 bps
Median slippage: -4.32 bps
Standard deviation: 1.88 bps
Slippage vs Order Size correlation: 0.026
Slippage vs Volatility correlation: -0.012
Slippage vs Liquidity correlation: 0.026
Visualizations saved to: results/images/
============================================================

========================================
STEP 5: LATENCY ANALYSIS FOR EOSUSDT
============================================================
COMPREHENSIVE LATENCY ANALYSIS
============================================================
Using processed data with 1516224 records.
Calculating latencies...
Calculated latencies for 1514 execution records
Generating latency summary statistics...

Generating visualizations...
Creating latency distribution plots...
Warning: Could not save PNG image. HTML version saved successfully.
Latency distribution plots saved to results/images/

Detecting latency outliers...
Found 46 exchange_processing_latency outliers (threshold: 2.5σ)
exchange_processing_latency outliers saved to results/eosusdt_exchange_processing_latency_outliers_1750867996.json
Found 8 network_latency outliers (threshold: 2.5σ)
network_latency outliers saved to results/eosusdt_network_latency_outliers_1750867996.json
Found 8 strategy_processing_latency outliers (threshold: 2.5σ)
strategy_processing_latency outliers saved to results/eosusdt_strategy_processing_latency_outliers_1750867996.json
============================================================

============================================================
LATENCY ANALYSIS SUMMARY
============================================================
Analysis completed in 0.13 seconds

Exchange Processing Latency:
  - Count: 1514
  - Mean: 18555.78 μs
  - Median: 3335.50 μs
  - P95: 80174.80 μs
  - P99: 187079.47 μs
  - Outliers (>2.5σ): 46 (3.04%)

Network Latency:
  - Count: 1514
  - Mean: 22.51 μs
  - Median: 20.00 μs
  - P95: 36.00 μs
  - P99: 56.00 μs
  - Outliers (>2.5σ): 8 (0.53%)

Strategy Processing Latency:
  - Count: 1514
  - Mean: 34290.62 μs
  - Median: 21000.00 μs
  - P95: 108350.00 μs
  - P99: 219180.00 μs
  - Outliers (>2.5σ): 8 (0.53%)

Visualization saved to: results/images/eosusdt_latency_distributions.html
Outlier files saved with prefix: eosusdt_
============================================================

========================================
STEP 6: FEE ANALYSIS FOR EOSUSDT
============================================================
COMPREHENSIVE FEE ANALYSIS
============================================================
Using processed data with 1516224 records.
Preparing fee analysis data...
Prepared fee analysis data for 1514 execution records
Detecting fee outliers with 2.0 sigma threshold...
Found 59 fee outliers out of 1514 records (3.90%)
Generating fee summary statistics...

Generating visualizations...
Creating fee distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee distribution plot saved to results/images/
Creating fee vs quantity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs quantity plot saved to results/images/
Creating fee vs VWAP plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs VWAP plot saved to results/images/
Creating fee vs notional value plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs notional value plot saved to results/images/
Analyzing fee outliers...
Fee outlier analysis exported to results/eosusdt_fee_outliers_20250626_001317.json

============================================================
FEE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.32 seconds
Total execution records analyzed: 1514
Fee statistics:
  - Mean: -0.006830
  - Median: -0.004656
  - Std: 0.007129
Correlations:
  - Fee vs Quantity: -0.893
  - Fee vs VWAP: -0.089
  - Fee vs Notional: -0.894
Outlier analysis:
  - Outliers found: 59 (3.9%)
  - Mean outlier fee: -0.026211
  - Mean normal fee: -0.006044
Visualizations saved to: results/images/
============================================================

========================================
STEP 7: SPREAD ANALYSIS FOR EOSUSDT
============================================================
COMPREHENSIVE SPREAD ANALYSIS
============================================================
Using processed data with 1516224 records.
Calculating bid and ask spreads...
Spread calculations completed
Preparing data for plotting...
Market data records: 1516224
Execution records: 1514
Generating summary statistics...

Generating visualizations...
Creating spread visualization...
Downsampling market data for visualization (interval: 100)...
Original market data points: 1516224, Sampled points: 15163
Warning: Could not save PNG image. HTML version saved successfully.
Visualization saved to results\images\eosusdt_spread_execution_visualization.png
Interactive HTML saved to results\images\eosusdt_spread_execution_visualization.html
Creating bid-ask spread distribution chart...
Market data: 1514710 records, 2801 (0.2%) exceed 99% percentile
Execution data: 1514 records, 7 (0.5%) exceed 99% percentile
99% percentile value: 0.001000
Warning: Could not save PNG image. HTML version saved successfully.
Bid-ask spread distribution saved to results\images\eosusdt_bidask_spread_distribution.png
Interactive HTML saved to results\images\eosusdt_bidask_spread_distribution.html
Exporting trades with bid-ask spread > 0.2 USDT...
No trades found with bid-ask spread > 0.2 USDT

============================================================
SPREAD ANALYSIS SUMMARY
============================================================
Analysis completed in 0.42 seconds
Market data records: 1516224
Execution records: 1514
Bid-ask spread (mean/median): 0.001002 / 0.001000 USDT
Buy orders: 736, Sell orders: 778
Visualizations saved to: results/images/
============================================================

============================================================
EOSUSDT PIPELINE EXECUTION SUMMARY
============================================================
Execution time: 1.64 seconds
Data preparation: LOADED_FOR_ANALYSIS
Market impact analysis: COMPLETE
Slippage analysis: COMPLETE
Latency analysis: COMPLETE
Fee analysis: COMPLETE
Spread analysis: COMPLETE

============================================================
PROCESSING ETHUSDT
============================================================

========================================
STEP 2: DATA PREPARATION FOR ETHUSDT
Processed data found at results/ethusdt_processed_data.parquet
Loading existing processed data for ETHUSDT analysis...
Loading processed data from results/ethusdt_processed_data.parquet...
Loaded 2461236 records in 0.44 seconds

========================================
STEP 3: MARKET IMPACT ANALYSIS FOR ETHUSDT

=== Market Impact Analysis ===

=== Market Impact Data Validation ===
eqty violations (sde=2, eqty > bidVol-1): 15
eqty violations (sde=1, eqty > askVol-1): 14
vwap violations (sde=2, vwap < bidPrice-1): 10
vwap violations (sde=1, vwap > askPrice-1): 6
Market Impact Data Validation Complete.

========================================
STEP 4: SLIPPAGE ANALYSIS FOR ETHUSDT
============================================================
COMPREHENSIVE SLIPPAGE ANALYSIS
============================================================
Using processed data with 2461236 records.
Calculating slippage...
Calculated slippage for 4508 execution records
Calculating market volatility with 100-period window...
Calculating liquidity measure with 10-period average...
Generating slippage summary statistics...

Generating visualizations...
Creating slippage distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage distribution plot saved to results/images/
Creating slippage vs order size plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs order size plot saved to results/images/
Creating slippage vs volatility plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs volatility plot saved to results/images/
Creating slippage vs liquidity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs liquidity plot saved to results/images/

============================================================
SLIPPAGE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.66 seconds
Total execution records analyzed: 4508
Mean slippage: -0.49 bps
Median slippage: -0.10 bps
Standard deviation: 0.70 bps
Slippage vs Order Size correlation: 0.001
Slippage vs Volatility correlation: -0.238
Slippage vs Liquidity correlation: 0.012
Visualizations saved to: results/images/
============================================================

========================================
STEP 5: LATENCY ANALYSIS FOR ETHUSDT
============================================================
COMPREHENSIVE LATENCY ANALYSIS
============================================================
Using processed data with 2461236 records.
Calculating latencies...
Calculated latencies for 4508 execution records
Generating latency summary statistics...

Generating visualizations...
Creating latency distribution plots...
Warning: Could not save PNG image. HTML version saved successfully.
Latency distribution plots saved to results/images/

Detecting latency outliers...
Found 47 exchange_processing_latency outliers (threshold: 2.5σ)
exchange_processing_latency outliers saved to results/ethusdt_exchange_processing_latency_outliers_1750867998.json
Found 104 network_latency outliers (threshold: 2.5σ)
network_latency outliers saved to results/ethusdt_network_latency_outliers_1750867998.json
Found 69 strategy_processing_latency outliers (threshold: 2.5σ)
strategy_processing_latency outliers saved to results/ethusdt_strategy_processing_latency_outliers_1750867998.json
============================================================

============================================================
LATENCY ANALYSIS SUMMARY
============================================================
Analysis completed in 0.14 seconds

Exchange Processing Latency:
  - Count: 4508
  - Mean: 10693.35 μs
  - Median: 2653.00 μs
  - P95: 28831.00 μs
  - P99: 51825.79 μs
  - Outliers (>2.5σ): 47 (1.04%)

Network Latency:
  - Count: 4508
  - Mean: 148.81 μs
  - Median: 98.00 μs
  - P95: 405.65 μs
  - P99: 770.00 μs
  - Outliers (>2.5σ): 104 (2.31%)

Strategy Processing Latency:
  - Count: 4508
  - Mean: 15791.93 μs
  - Median: 15000.00 μs
  - P95: 33000.00 μs
  - P99: 49000.00 μs
  - Outliers (>2.5σ): 69 (1.53%)

Visualization saved to: results/images/ethusdt_latency_distributions.html
Outlier files saved with prefix: ethusdt_
============================================================

========================================
STEP 6: FEE ANALYSIS FOR ETHUSDT
============================================================
COMPREHENSIVE FEE ANALYSIS
============================================================
Using processed data with 2461236 records.
Preparing fee analysis data...
Prepared fee analysis data for 4508 execution records
Detecting fee outliers with 2.0 sigma threshold...
Found 101 fee outliers out of 4508 records (2.24%)
Generating fee summary statistics...

Generating visualizations...
Creating fee distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee distribution plot saved to results/images/
Creating fee vs quantity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs quantity plot saved to results/images/
Creating fee vs VWAP plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs VWAP plot saved to results/images/
Creating fee vs notional value plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs notional value plot saved to results/images/
Analyzing fee outliers...
Fee outlier analysis exported to results/ethusdt_fee_outliers_20250626_001319.json

============================================================
FEE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.31 seconds
Total execution records analyzed: 4508
Fee statistics:
  - Mean: -0.003018
  - Median: -0.003093
  - Std: 0.001677
Correlations:
  - Fee vs Quantity: -1.000
  - Fee vs VWAP: -0.087
  - Fee vs Notional: -1.000
Outlier analysis:
  - Outliers found: 101 (2.2%)
  - Mean outlier fee: -0.008186
  - Mean normal fee: -0.002900
Visualizations saved to: results/images/
============================================================

========================================
STEP 7: SPREAD ANALYSIS FOR ETHUSDT
============================================================
COMPREHENSIVE SPREAD ANALYSIS
============================================================
Using processed data with 2461236 records.
Calculating bid and ask spreads...
Spread calculations completed
Preparing data for plotting...
Market data records: 2461236
Execution records: 4508
Generating summary statistics...

Generating visualizations...
Creating spread visualization...
Downsampling market data for visualization (interval: 100)...
Original market data points: 2461236, Sampled points: 24613
Warning: Could not save PNG image. HTML version saved successfully.
Visualization saved to results\images\ethusdt_spread_execution_visualization.png
Interactive HTML saved to results\images\ethusdt_spread_execution_visualization.html
Creating bid-ask spread distribution chart...
Market data: 2456728 records, 23942 (1.0%) exceed 99% percentile
Execution data: 4508 records, 79 (1.8%) exceed 99% percentile
99% percentile value: 0.160000
Warning: Could not save PNG image. HTML version saved successfully.
Bid-ask spread distribution saved to results\images\ethusdt_bidask_spread_distribution.png
Interactive HTML saved to results\images\ethusdt_bidask_spread_distribution.html
Exporting trades with bid-ask spread > 0.2 USDT...
Exported 45 trades with large spreads to results\ethusdt_large_spread_trades_0.2usdt_20250626_001319.json

============================================================
SPREAD ANALYSIS SUMMARY
============================================================
Analysis completed in 0.63 seconds
Market data records: 2461236
Execution records: 4508
Bid-ask spread (mean/median): 0.015418 / 0.010000 USDT
Buy orders: 2677, Sell orders: 1831
Visualizations saved to: results/images/
============================================================

============================================================
ETHUSDT PIPELINE EXECUTION SUMMARY
============================================================
Execution time: 2.28 seconds
Data preparation: LOADED_FOR_ANALYSIS
Market impact analysis: COMPLETE
Slippage analysis: COMPLETE
Latency analysis: COMPLETE
Fee analysis: COMPLETE
Spread analysis: COMPLETE

============================================================
PROCESSING XRPUSDT
============================================================

========================================
STEP 2: DATA PREPARATION FOR XRPUSDT
Processed data found at results/xrpusdt_processed_data.parquet
Loading existing processed data for XRPUSDT analysis...
Loading processed data from results/xrpusdt_processed_data.parquet...
Loaded 1889199 records in 0.21 seconds

========================================
STEP 3: MARKET IMPACT ANALYSIS FOR XRPUSDT

=== Market Impact Analysis ===

=== Market Impact Data Validation ===
eqty violations (sde=2, eqty > bidVol-1): 0
eqty violations (sde=1, eqty > askVol-1): 1
vwap violations (sde=2, vwap < bidPrice-1): 0
vwap violations (sde=1, vwap > askPrice-1): 0
Market Impact Data Validation Complete.

========================================
STEP 4: SLIPPAGE ANALYSIS FOR XRPUSDT
============================================================
COMPREHENSIVE SLIPPAGE ANALYSIS
============================================================
Using processed data with 1889199 records.
Calculating slippage...
Calculated slippage for 2492 execution records
Calculating market volatility with 100-period window...
Calculating liquidity measure with 10-period average...
Generating slippage summary statistics...

Generating visualizations...
Creating slippage distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage distribution plot saved to results/images/
Creating slippage vs order size plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs order size plot saved to results/images/
Creating slippage vs volatility plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs volatility plot saved to results/images/
Creating slippage vs liquidity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Slippage vs liquidity plot saved to results/images/

============================================================
SLIPPAGE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.57 seconds
Total execution records analyzed: 2492
Mean slippage: -1.57 bps
Median slippage: -1.45 bps
Standard deviation: 0.63 bps
Slippage vs Order Size correlation: 0.006
Slippage vs Volatility correlation: -0.075
Slippage vs Liquidity correlation: 0.040
Visualizations saved to: results/images/
============================================================

========================================
STEP 5: LATENCY ANALYSIS FOR XRPUSDT
============================================================
COMPREHENSIVE LATENCY ANALYSIS
============================================================
Using processed data with 1889199 records.
Calculating latencies...
Calculated latencies for 2492 execution records
Generating latency summary statistics...

Generating visualizations...
Creating latency distribution plots...
Warning: Could not save PNG image. HTML version saved successfully.
Latency distribution plots saved to results/images/

Detecting latency outliers...
Found 41 exchange_processing_latency outliers (threshold: 2.5σ)
exchange_processing_latency outliers saved to results/xrpusdt_exchange_processing_latency_outliers_1750868000.json
Found 39 network_latency outliers (threshold: 2.5σ)
network_latency outliers saved to results/xrpusdt_network_latency_outliers_1750868000.json
Found 39 strategy_processing_latency outliers (threshold: 2.5σ)
strategy_processing_latency outliers saved to results/xrpusdt_strategy_processing_latency_outliers_1750868000.json
============================================================

============================================================
LATENCY ANALYSIS SUMMARY
============================================================
Analysis completed in 0.14 seconds

Exchange Processing Latency:
  - Count: 2492
  - Mean: 14961.96 μs
  - Median: 2820.00 μs
  - P95: 49756.40 μs
  - P99: 118828.95 μs
  - Outliers (>2.5σ): 41 (1.65%)

Network Latency:
  - Count: 2492
  - Mean: 21.73 μs
  - Median: 20.00 μs
  - P95: 37.00 μs
  - P99: 59.09 μs
  - Outliers (>2.5σ): 39 (1.57%)

Strategy Processing Latency:
  - Count: 2492
  - Mean: 23362.76 μs
  - Median: 19000.00 μs
  - P95: 57000.00 μs
  - P99: 118270.00 μs
  - Outliers (>2.5σ): 39 (1.57%)

Visualization saved to: results/images/xrpusdt_latency_distributions.html
Outlier files saved with prefix: xrpusdt_
============================================================

========================================
STEP 6: FEE ANALYSIS FOR XRPUSDT
============================================================
COMPREHENSIVE FEE ANALYSIS
============================================================
Using processed data with 1889199 records.
Preparing fee analysis data...
Prepared fee analysis data for 2492 execution records
Detecting fee outliers with 2.0 sigma threshold...
Found 110 fee outliers out of 2492 records (4.41%)
Generating fee summary statistics...

Generating visualizations...
Creating fee distribution plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee distribution plot saved to results/images/
Creating fee vs quantity plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs quantity plot saved to results/images/
Creating fee vs VWAP plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs VWAP plot saved to results/images/
Creating fee vs notional value plot...
Warning: Could not save PNG image. HTML version saved successfully.
Fee vs notional value plot saved to results/images/
Analyzing fee outliers...
Fee outlier analysis exported to results/xrpusdt_fee_outliers_20250626_001321.json

============================================================
FEE ANALYSIS SUMMARY
============================================================
Analysis completed in 0.31 seconds
Total execution records analyzed: 2492
Fee statistics:
  - Mean: -0.004415
  - Median: -0.003144
  - Std: 0.003778
Correlations:
  - Fee vs Quantity: -1.000
  - Fee vs VWAP: -0.091
  - Fee vs Notional: -1.000
Outlier analysis:
  - Outliers found: 110 (4.4%)
  - Mean outlier fee: -0.016942
  - Mean normal fee: -0.003837
Visualizations saved to: results/images/
============================================================

========================================
STEP 7: SPREAD ANALYSIS FOR XRPUSDT
============================================================
COMPREHENSIVE SPREAD ANALYSIS
============================================================
Using processed data with 1889199 records.
Calculating bid and ask spreads...
Spread calculations completed
Preparing data for plotting...
Market data records: 1889199
Execution records: 2492
Generating summary statistics...

Generating visualizations...
Creating spread visualization...
Downsampling market data for visualization (interval: 100)...
Original market data points: 1889199, Sampled points: 18892
Warning: Could not save PNG image. HTML version saved successfully.
Visualization saved to results\images\xrpusdt_spread_execution_visualization.png
Interactive HTML saved to results\images\xrpusdt_spread_execution_visualization.html
Creating bid-ask spread distribution chart...
Market data: 1886707 records, 7576 (0.4%) exceed 99% percentile
Execution data: 2492 records, 24 (1.0%) exceed 99% percentile
99% percentile value: 0.000100
Warning: Could not save PNG image. HTML version saved successfully.
Bid-ask spread distribution saved to results\images\xrpusdt_bidask_spread_distribution.png
Interactive HTML saved to results\images\xrpusdt_bidask_spread_distribution.html
Exporting trades with bid-ask spread > 0.2 USDT...
No trades found with bid-ask spread > 0.2 USDT

============================================================
SPREAD ANALYSIS SUMMARY
============================================================
Analysis completed in 0.50 seconds
Market data records: 1889199
Execution records: 2492
Bid-ask spread (mean/median): 0.000100 / 0.000100 USDT
Buy orders: 648, Sell orders: 1844
Visualizations saved to: results/images/
============================================================

============================================================
XRPUSDT PIPELINE EXECUTION SUMMARY
============================================================
Execution time: 1.81 seconds
Data preparation: LOADED_FOR_ANALYSIS
Market impact analysis: COMPLETE
Slippage analysis: COMPLETE
Latency analysis: COMPLETE
Fee analysis: COMPLETE
Spread analysis: COMPLETE

============================================================
OVERALL PIPELINE EXECUTION SUMMARY
============================================================
Total execution time: 12.44 seconds
Symbols processed: 5
Successful: 5
Failed: 0
Data validation: COMPLETE

BTCUSDT:
  Status: COMPLETE
  Execution time: 2.50 seconds
  Market impact violations: 47
  Slippage (mean): -0.21 bps

DOTUSDT:
  Status: COMPLETE
  Execution time: 1.87 seconds
  Market impact violations: 326
  Slippage (mean): -1.04 bps

EOSUSDT:
  Status: COMPLETE
  Execution time: 1.64 seconds
  Market impact violations: 20
  Slippage (mean): -4.55 bps

ETHUSDT:
  Status: COMPLETE
  Execution time: 2.28 seconds
  Market impact violations: 428
  Slippage (mean): -0.49 bps

XRPUSDT:
  Status: COMPLETE
  Execution time: 1.81 seconds
  Market impact violations: 46
  Slippage (mean): -1.57 bps
============================================================
Pipeline completed successfully!
