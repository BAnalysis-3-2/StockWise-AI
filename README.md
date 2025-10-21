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

<h3>1. User Uploads Sales Data</h3>
<p><strong>Format:</strong> CSV file (<code>retail_store_inventory.csv</code>)</p>
<p><strong>Columns:</strong> product id, price, discount, inventory, weather, holiday, etc.</p>

<h3>2. Data Preprocessing (<code>preprocess_data.py</code>)</h3>
<ul>
  <li>Cleans and normalizes columns</li>
  <li>Constructs <code>sku</code> from <code>store_id + product_id</code></li>
  <li>Sorts by SKU and date</li>
  <li>Saves:
    <ul>
      <li><code>training_data.csv</code>: cleaned raw data</li>
      <li><code>training_features.csv</code>: engineered features</li>
    </ul>
  </li>
  <li><strong>Feature Engineering Includes:</strong>
    <ul>
      <li>Lag features: <code>lag_1</code>, <code>lag_7</code>, <code>lag_14</code></li>
      <li>Rolling stats: <code>rolling_mean_3</code>, <code>rolling_mean_7</code>, <code>rolling_std_7</code></li>
      <li>Temporal: <code>day_of_week</code>, <code>month</code>, <code>year</code></li>
      <li>Price change, discount flag</li>
    </ul>
  </li>
</ul>

<h3>3. Model Routing and Forecasting (<code>sku_forecaster.py</code>)</h3>
<p>Each SKU is routed based on row count:</p>
<table>
  <thead>
    <tr>
      <th>Row Count</th>
      <th>Model Used</th>
      <th>Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><30</td>
      <td>ARIMA</td>
      <td>Time series</td>
    </tr>
    <tr>
      <td>>=30</td>
      <td>LSTM</td>
      <td>Deep learning</td>
    </tr>
  </tbody>
</table>
<p><strong>Output:</strong> <code>sku_forecast.csv</code> with columns: <code>sku</code>, <code>date</code>, <code>forecast</code>, <code>model_used</code>, <code>mae</code></p>

<h3>4. Chatbot Interface</h3>
<p>Users interact via a terminal menu:</p>
<ul>
  <li>1 – View forecast for a SKU</li>
  <li>2 – Compare forecasts between SKUs</li>
  <li>4 – Show top SKUs by forecast</li>
  <li>5 – Ask a custom question (NLP)</li>
  <li>6 – Upload your own sales CSV and run forecast</li>
  <li>7 – Simulate a demand scenario</li>
  <li>8 – Exit</li>
</ul>

<h3>5. Simulation Engine (<code>simulate_scenario.py</code>)</h3>
<p><strong>User Inputs:</strong> SKU, price, discount, inventory, holiday flag</p>
<p><strong>System Actions:</strong></p>
<ul>
  <li>Loads last known feature row for that SKU</li>
  <li>Injects new values</li>
  <li>Runs prediction using:
    <ul>
      <li>LightGBM (<code>lgb_model.pkl</code>)</li>
      <li>XGBoost (<code>xgb_model.pkl</code>)</li>
    </ul>
  </li>
<h5>Outputs simulated demand from both models</h5>
</ul>

<h3>Training Data</h3>
<pre>
date,store_id,product_id,category,region,inventory_level,units_sold,units_ordered,demand_forecast,price,discount,weather_condition,holiday/promotion,competitor_pricing,seasonality
2022/01/01,S001,P0001,Groceries,North,231,127,55,135.47,33.5,20,Rainy,0,29.69,Autumn
2022/01/01,S001,P0002,Toys,South,204,150,66,144.04,63.01,20,Sunny,0,66.16,Autumn
2022/01/01,S001,P0003,Toys,West,102,65,51,74.02,27.99,10,Sunny,1,31.32,Summer
2022/01/01,S001,P0004,Toys,North,469,61,164,62.18,32.72,10,Cloudy,1,34.74,Autumn
...
</pre>
