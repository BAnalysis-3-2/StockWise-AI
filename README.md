<h1>AI Demand Forecasting System</h1>

<h2>Problem Definition</h2>
<p>
Small and medium-sized enterprises (SMEs) in South Africa often struggle with inventory mismanagement due to inaccurate demand forecasts. This project addresses that challenge with an AI-driven solution aligned with the theme of <strong>AI Solutions for Industry</strong>. The system improves decision-making, reduces waste, and enhances operational efficiency.
</p>

<h2>AI Solution</h2>
<p>
This system forecasts retail demand using strict data thresholds and forensic-grade preprocessing. It routes SKUs to ARIMA or LSTM based on row count and produces transparent, teachable outputs for stakeholder review.
</p>
<p>
It leverages a hybrid AI architecture combining multiple data sources—retail sales, economic indicators, and social media sentiment—with advanced models including LightGBM, XGBoost, ARIMA, and LSTM. The system supports real-time alerts and stakeholder interaction via an integrated chatbot interface.
</p>

<h2>Business Background</h2>
<p>
Inventory mismanagement limits SME scalability and profitability. This project supports South Africa’s 4IR goals by deploying AI-powered forecasting tools to enhance productivity, decision-making, and operational agility. It empowers SMEs to compete more effectively in modern markets.
</p>

<h2>Key Features</h2>
<ul>
  <li>Modular preprocessing with strict column validation</li>
  <li>Feature engineering: lag, rolling stats, price change, inventory ratio</li>
  <li>Model routing: LSTM for 30 or more rows and ARIMA for less than 30 rows</li>
  <li>Forecast output: SKU-level CSV with model_used and MAE</li>
  <li>Chatbot interface for forecast lookup and simulation</li>
</ul>

<h2>Tools and Techniques</h2>
<ul>
  <li><strong>Machine Learning Platform:</strong> Python with scikit-learn, LightGBM, XGBoost, TensorFlow/Keras</li>
  <li><strong>Time-Series Forecasting:</strong> ARIMA, Long Short-Term Memory (LSTM)</li>
  <li><strong>Data Processing & Integration:</strong> Python libraries for feature extraction, sentiment analysis, and data cleaning</li>
</ul>

<h2>Machine Learning Approach</h2>
<ul>
  <li><strong>Ensemble Modeling:</strong> Combines LightGBM, XGBoost, ARIMA, and LSTM for robust forecasts</li>
  <li><strong>Transfer & Few-Shot Learning:</strong> Adapts to new stores/products with limited historical data</li>
  <li><strong>Automated Feature Extraction:</strong> Derives lag variables, seasonal indicators, and sentiment features</li>
  <li><strong>Performance Monitoring & Retraining:</strong> Triggers model updates based on drift detection</li>
</ul>

<h3>How It Learns</h3>
<p>
The system processes sequences of historical sales data to uncover complex temporal dependencies. LSTM models learn long-term patterns that traditional time-series models may miss, improving forecast accuracy.
</p>

<h2>Constraints</h2>
<ul>
  <li><strong>Data Quality:</strong> Incomplete or inconsistent sales data may impact accuracy. Rigorous validation is enforced.</li>
  <li><strong>User Adoption:</strong> Retailers accustomed to manual processes may require training and change management.</li>
  <li><strong>Technical Integration:</strong> Legacy POS/ERP systems are supported via modular API design.</li>
</ul>

<h2>Requirements</h2>
<ul>
  <li>Python 3.10</li>
  <li>TensorFlow</li>
  <li>scikit-learn, LightGBM, XGBoost</li>
  <li>pandas, numpy, matplotlib, seaborn</li>
</ul>

<h2>Installation</h2>
<pre>
git clone https://github.com/your-username/demand-forecasting-engine.git
cd demand-forecasting-engine
\venv\Scripts\activate        # activate environment
pip install -r requirements.txt
</pre>
<p>
Install the <code>requirements.txt</code> file to ensure correct versions of libraries are used.
</p>

<h2>How the System Works</h2>

<h4>1. User Uploads Sales Data</h4>
Format: CSV file (retail_store_inventory.csv)
columns: product id, price, discount, inventory level, weather, holiday, etc.

<h4>2. Data Preprocessing (preprocess_data.py)</h4>
- Cleans and normalizes columns
- Constructs sku from store_id + product_id
- Sorts by SKU and date

Saves:
- training_data.csv: cleaned raw data
- training_features.csv: engineered features

Feature Engineering Includes:
- Lag features: lag_1, lag_7, lag_14
- Rolling stats: rolling_mean_3, rolling_mean_7, rolling_std_7
- Temporal: day_of_week, month, year
- Price change, discount flag

<h4>3. Model Routing and Forecasting (sku_forecaster.py)</h4>

Each SKU is routed based on row count:
- <30	ARIMA	Time series
- ≥30	LSTM	Deep learning

Forecasts are saved to sku_forecast.csv, with columns:
- sku, date, forecast, model_used, mae

<h4>4. Chatbot Interface</h4>

Users interact via a terminal menu:

Option	Function
1	View forecast for a SKU
2	Compare forecasts between SKUs
4	Show top SKUs by forecast
5	Ask a custom question (NLP)
6	Upload your own sales CSV and run forecast
7	Simulate a demand scenario
8	Exit

<h4>5. Simulation Engine (simulate_scenario.py)</h4>

User inputs:
- SKU, price, discount, inventory, holiday flag

System:
- Loads last known feature row for that SKU
- Injects new values

Runs prediction using:
- LightGBM (lgb_model.pkl)
- XGBoost (xgb_model.pkl)

Outputs simulated demand from both models


<h3>Training Data</h3>
<pre>
date,store_id,product_id,category,region,inventory_level,units_sold,units_ordered,demand_forecast,price,discount,weather_condition,holiday/promotion,competitor_pricing,seasonality
2022/01/01,S001,P0001,Groceries,North,231,127,55,135.47,33.5,20,Rainy,0,29.69,Autumn
2022/01/01,S001,P0002,Toys,South,204,150,66,144.04,63.01,20,Sunny,0,66.16,Autumn
2022/01/01,S001,P0003,Toys,West,102,65,51,74.02,27.99,10,Sunny,1,31.32,Summer
2022/01/01,S001,P0004,Toys,North,469,61,164,62.18,32.72,10,Cloudy,1,34.74,Autumn
...
</pre>
