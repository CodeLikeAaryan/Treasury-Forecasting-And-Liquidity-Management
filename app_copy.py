from flask import Flask, request, jsonify, render_template, flash, redirect
import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, jsonify
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import os
import io
import base64
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.io as pio
from prophet import Prophet

app = Flask(__name__)

# Global variable for customer data
customers_list = None

# Path to the static folder where pre-generated plots are stored
static_dir = "static"

def set_professional_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight='bold', color='#34495e', pad=15)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='normal', color='#7f8c8d')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal', color='#7f8c8d')
    ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='#2c3e50')
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, color='#bdc3c7', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, color='#ecf0f1', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#95a5a6')
        spine.set_linewidth(1.2)
        spine.set_capstyle('round')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Load customer data
def load_customer_data():
    global customers_list
    customers_list = [
        "ALLEGIANT NETWORKS TM", "Asit SpA", "AT&T CORP", "AVAD GmbH", 
        "CommsPlus Distribution", "FIORE S.R.L. PROFESSIONALE", "Itancia", 
        "Itancia IPO Cloud", "JENNE DISTRIBUTORS INC SMBS", "Keyston Distribution FZCO", 
        "Mauritius Telecom", "OLIBRA TELECOMUNICACAO", 
        "SCANSOURCE INC DBA CATALYST TELECOM", "TD Synnex Canada ULC", 
        "VOIP Distribution Pty Ltd", "Westcon Group Netherlands B.V."
    ]
load_customer_data()
# Load the data for insights
file_path = 'filtered_ap.csv'
df = pd.read_csv(file_path)
df = df.dropna(subset=['Supplier or Party'])


# Load the model
model = pickle.load(open('Cols_xgb_model_T5.pkl', 'rb'))

# Load the models and scalers
ar_scaler = pickle.load(open('scaler_xgb_ar_final.pkl', 'rb'))
ar_model = pickle.load(open('xgb_model_ar_final.pkl', 'rb'))
ap_scaler = pickle.load(open('scaler_xgb_ap.pkl', 'rb'))
ap_model = pickle.load(open('xgb_model_ap.pkl', 'rb'))

# Load datasets
ar_data = pd.read_csv('finalll_ar.csv', parse_dates=['Payment date'])
ap_data = pd.read_csv('finall_ap.csv', parse_dates=['Payment date'])

# Custom accuracy function (within a threshold)
def regression_accuracy(y_true, y_pred, threshold=0.05):
    accuracy = np.mean(np.abs(y_pred - y_true) / np.abs(y_true) <= threshold)
    return accuracy * 100  # Return as percentage

def create_due_date(row):
    try:
        year, month, day = int(row['Due_Year']), int(row['Due_Month']), int(row['Due_Day'])
        if month > 0 and month <= 12 and day > 0 and day <= 31:
            return pd.Timestamp(year=year, month=month, day=day)
        else:
            return pd.NaT  
    except ValueError:
        return pd.NaT  

def make_forecast(model, data, scaler, days):
    last_data = data['Amount'].values[-1].reshape(-1, 1)
    scaled_last_data = scaler.transform(last_data)
    forecast = []
    input_data = scaled_last_data

    for _ in range(days):
        prediction = model.predict(input_data)
        forecast.append(prediction[0])
        input_data = np.roll(input_data, -1) 
        input_data[-1] = prediction  

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten()


@app.route('/')
def index():
    return render_template('index.html')
def plot_to_img_tag(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return f'<img src="data:image/png;base64,{img_data}" style="width:100%;"/>'

@app.route('/payment-date-prediction', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            df.to_csv('uploaded.csv', index=False)  # Temporarily save the file to use later
            return render_template('payment-date-prediction.html', uploaded=True)
    return render_template('payment-date-prediction.html', uploaded=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df.to_csv('uploaded.csv', index=False)
        return jsonify({'success': True, 'message': 'File has been uploaded'})
    return jsonify({'success': False, 'message': 'Unsupported file type'})

@app.route('/view-data', methods=['POST'])
def view_data():
    try:
        df = pd.read_csv('uploaded.csv')
        df_top10 = df.head(10)  # Select only the top 10 rows
        return df_top10.to_html(classes='data')
    except Exception as e:
        return str(e), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the uploaded CSV file
        df = pd.read_csv('uploaded.csv')

        # Ensure the model's feature columns match the CSV
        feature_columns = ['Due_Year', 'BLINE_Year', 'AGING', 'Due in Week Number', 'BLINE_WeekOfYear', 
                           'BLINE_Day', 'Due_Month', 'Standard Amount in USD', 'Document_Year', 
                           'Due_WeekOfYear', 'BLINE_Month', 'Payment Terms in Days', 
                           'Due in Month number', 'Due_Day', 'DaysPastBLINE', 'DaysToDue', 
                           'Document_WeekOfYear']
        if not all(col in df.columns for col in feature_columns):
            return "Uploaded CSV does not have the required columns.", 400

        df_features = df[feature_columns]

        # Make predictions (assuming `model` and `create_due_date` are predefined in your environment)
        predictions = model.predict(df_features)

        # Add the predictions as a new column in the DataFrame
        df['Predicted Days in Arrears'] = predictions

        # Round "Predicted Days in Arrears" to the nearest integer
        df['Predicted Days in Arrears'] = df['Predicted Days in Arrears'].round(0).astype(int)

        # Apply the create_due_date function to safely generate the 'Due Date' column
        df['Due Date'] = df.apply(create_due_date, axis=1)

        # Calculate 'Predicted Payment Receive Date' as Due Date + Predicted Days in Arrears
        df['Predicted Payment Receive Date'] = df['Due Date'] + pd.to_timedelta(df['Predicted Days in Arrears'], unit='D')

        # Convert "Predicted Payment Receive Date" to date format (drop the time part)
        df['Predicted Payment Receive Date'] = df['Predicted Payment Receive Date'].dt.date

        # Round "Standard Amount in USD" and convert to integer
        df['Standard Amount in USD'] = df['Standard Amount in USD'].round(0).astype(int)

        # Create the 'Date Bucket' column based on the 'Predicted Days in Arrears' value
        def assign_date_bucket(days_in_arrears):
            if days_in_arrears < 0:
                return 'Less than 0'
            elif 0 <= days_in_arrears <= 15:
                return '0-15'
            elif 16 <= days_in_arrears <= 30:
                return '16-30'
            elif 31 <= days_in_arrears <= 45:
                return '31-45'
            else:
                return '45+'

        df['Date Bucket'] = df['Predicted Days in Arrears'].apply(assign_date_bucket)

   
        def assign_payer_type(row):
            if row['Predicted Days in Arrears'] < 0:
                return 'Early Payer'
            elif 0 <= row['Predicted Days in Arrears'] <= row['Payment Terms in Days']:
                return 'On-Time Payer'
            else:
                return 'Late Payer'

        df['Payer Type'] = df.apply(assign_payer_type, axis=1)

        result_df = df[['Customer Name','Due Date', 'Predicted Days in Arrears', 'Predicted Payment Receive Date', 
                        'Standard Amount in USD', 'Payment Terms in Days', 'Date Bucket', 'Payer Type']]
        result_html = result_df.to_html(classes='data')
        df.to_csv('predicted_data.csv', index=False)
        unique_customers = result_df['Customer Name'].unique().tolist()
        return jsonify({
            'table': result_html,
            'customers': unique_customers  
        })

    except Exception as e:
        return str(e), 400


@app.route('/insights', methods=['GET'])
def generate_customer_insights():
    selected_customer = request.args.get('customer')
    df = pd.read_csv('predicted_data.csv')
    customer_df = df[df['Customer Name'] == selected_customer]

    if customer_df.empty:
        return "No data available for the selected customer."

    # Calculate insights
    total_transactions = customer_df.shape[0]
    average_days_in_arrears = customer_df['Predicted Days in Arrears'].mean()
    total_amount_transacted = customer_df['Standard Amount in USD'].sum()
    max_days_in_arrears = customer_df['Predicted Days in Arrears'].max()
    min_days_in_arrears = customer_df['Predicted Days in Arrears'].min()
    average_amount_transacted = customer_df['Standard Amount in USD'].mean()

    # Payments within term based on Payment Terms in Days
    payments_within_term = customer_df[customer_df['Predicted Days in Arrears'] <= customer_df['Payment Terms in Days']].shape[0]

    # Percentage of late payments
    late_payers_count = customer_df[customer_df['Predicted Days in Arrears'] > customer_df['Payment Terms in Days']].shape[0]
    late_payer_percentage = (late_payers_count / total_transactions) * 100 if total_transactions > 0 else 0

    # Aging analysis
    def assign_aging_bucket(days_in_arrears):
        if days_in_arrears <= 0:
            return 'Before Due'
        elif days_in_arrears <= 30:
            return '1-30 Days'
        elif days_in_arrears <= 60:
            return '31-60 Days'
        elif days_in_arrears <= 90:
            return '61-90 Days'
        else:
            return '>90 Days'

    customer_df['Aging Bucket'] = customer_df['Predicted Days in Arrears'].apply(assign_aging_bucket)

    aging_summary = customer_df.groupby('Aging Bucket')['Standard Amount in USD'].sum().reset_index()

    # Enhanced anomaly detection using statistical methods
    arrears_95th_percentile = customer_df['Predicted Days in Arrears'].quantile(0.95)
    payment_amount_95th_percentile = customer_df['Standard Amount in USD'].quantile(0.95)

    arrear_days_anomaly = customer_df[customer_df['Predicted Days in Arrears'] > arrears_95th_percentile].shape[0]
    payment_anomaly = customer_df[customer_df['Standard Amount in USD'] > payment_amount_95th_percentile].shape[0]

    # Customer Classification Logic
    if average_days_in_arrears <= 0 and late_payer_percentage == 0:
        customer_classification = 'Excellent Customer'
        classification_reason = (
            f"The customer consistently pays on time or even early, with an average days in arrears of "
            f"{average_days_in_arrears:.2f} days. They have a perfect record of no late payments (0%). "
            f"This behavior indicates a strong financial discipline and reliability."
        )
        risk_score = 5  # Low risk for excellent customers

    elif average_days_in_arrears <= 10 and late_payer_percentage <= 10:
        customer_classification = 'Good Customer'
        classification_reason = (
            f"The customer shows minor delays in payment, with an average days in arrears of "
            f"{average_days_in_arrears:.2f} days and a late payment percentage of {late_payer_percentage:.2f}%. "
            f"These delays are within acceptable limits, reflecting good financial behavior."
        )
        risk_score = 15  # Low to moderate risk for good customers

    elif average_days_in_arrears <= 30 and late_payer_percentage <= 30:
        customer_classification = 'Moderate Risk Customer'
        classification_reason = (
            f"The customer occasionally delays payments, with an average of {average_days_in_arrears:.2f} days in arrears "
            f"and a late payment percentage of {late_payer_percentage:.2f}%. While not critical, this behavior indicates "
            f"a moderate risk that may need attention."
        )
        risk_score = 50  # Moderate risk for moderate customers

    elif average_days_in_arrears > 90 or late_payer_percentage > 50:
        customer_classification = 'High-Risk Customer'
        classification_reason = (
            f"With an average of {average_days_in_arrears:.2f} days in arrears and a late payment percentage of "
            f"{late_payer_percentage:.2f}%, this customer poses a high risk. The significant delays in payment exceed 90 days, "
            f"indicating potential financial instability or default risk."
        )
        risk_score = 90  # High risk for high-risk customers

    else:
        customer_classification = 'Potential Risk Customer'
        classification_reason = (
            f"The customer has an average of {average_days_in_arrears:.2f} days in arrears and a late payment percentage of "
            f"{late_payer_percentage:.2f}%. While not immediately alarming, these figures warrant close monitoring as "
            f"they suggest an elevated risk of late payments."
        )
        risk_score = 75  # Elevated risk for potential risk customers

    # Generate the card layout as HTML dynamically
    insights_html = f"""
    <div class="card-container">
        <div class="card">
            <h4>Total Transactions</h4>
            <p>{total_transactions}</p>
            <span>Transactions</span>
        </div>
        <div class="card">
            <h4>Average Days in Arrears</h4>
            <p>{average_days_in_arrears:.2f}</p>
            <span>Days</span>
        </div>
        <div class="card">
            <h4>Total Amount Transacted</h4>
            <p>${total_amount_transacted:.2f}</p>
            <span>USD</span>
        </div>
        <div class="card">
            <h4>Max Days in Arrears</h4>
            <p>{max_days_in_arrears}</p>
            <span>Days</span>
        </div>
        <div class="card">
            <h4>Min Days in Arrears</h4>
            <p>{min_days_in_arrears}</p>
            <span>Days</span>
        </div>
        <div class="card">
            <h4>Average Amount Transacted</h4>
            <p>${average_amount_transacted:.2f}</p>
            <span>USD</span>
        </div>
        <div class="card">
            <h4>Payments Within Terms</h4>
            <p>{payments_within_term}</p>
            <span>Transactions</span>
        </div>
        <div class="card">
            <h4>Late Payment Percentage</h4>
            <p>{late_payer_percentage:.2f}%</p>
            <span>Percentage</span>
        </div>
        <div class="card">
            <h4>Arrear Days Anomalies</h4>
            <p>{arrear_days_anomaly}</p>
            <span>Transactions &gt; 95th Percentile</span>
        </div>
        <div class="card">
            <h4>Payment Amount Anomalies</h4>
            <p>{payment_anomaly}</p>
            <span>Transactions &gt; 95th Percentile</span>
        </div>
    </div>
    <div style="margin-top: 20px;"></div>
    """

    aging_html = aging_summary.to_html(index=False)
    # Initialize the summary list
    aging_bucket_summary = []

    total_amount = aging_summary['Standard Amount in USD'].sum()

    # Iterate through each row in the aging_summary dataframe
    for _, row in aging_summary.iterrows():
        bucket = row['Aging Bucket']
        amount = row['Standard Amount in USD']
        transaction_count = row.get('Transaction Count', 'N/A')  # Assuming there's a transaction count column
        
        if amount > 0:
            # Calculate the percentage contribution of the bucket to the total amount
            percentage_of_total = (amount / total_amount) * 100 if total_amount > 0 else 0
            aging_bucket_summary.append(
                f"For the <span style='font-weight: bold; color: #cc0d66;'>{bucket}</span> bucket, "
                f"the total amount is <span style='font-weight: bold;'>${amount:.2f}</span>, "
                f"which is <span style='font-weight: bold;'>{percentage_of_total:.2f}%</span> of the total amount."
            )
        else:
            aging_bucket_summary.append(
                f"No amount was received for the <span style='font-weight: bold; color: #cc0d66;'>{bucket}</span> bucket."
            )

    # Generate the final HTML description with matching styles
    aging_description = f"""
    <div class="aging-summary" style="background-color: #f8f8f8; padding: 20px; border-radius: 10px;">
        <h3 style="color: darkblue; font-weight: bold;">Aging Bucket Summary</h3>
        <p style="font-size: 21px; color: #333333;">
            {' '.join(aging_bucket_summary)}
        </p>
    </div>
    """

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={
            'text': f"<b>Customer Risk Classification:</b> {customer_classification}",  # Bold text for emphasis
            'font': {'size': 22, 'family': 'Arial', 'color': 'darkblue'}  # Improved font style and color
        },
        number={
            'font': {'size': 36, 'color': 'darkblue'}  # Larger font size for the risk score number
        },
        gauge={
    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "black"},  # Thicker ticks and darker color
    'bgcolor': "rgba(0,0,0,0)",  # Transparent background
    'borderwidth': 2,  # Subtle border around the gauge
    'bordercolor': "gray",  # Border color
    'steps': [  # Gradient color transition for each risk step
        {'range': [0, 20], 'color': "#2ECC71"},      # Green for low risk
        {'range': [20, 40], 'color': "#58D68D"},     # Light green
        {'range': [40, 60], 'color': "#F4D03F"},     # Yellow for moderate risk
        {'range': [60, 80], 'color': "#F39C12"},     # Orange for elevated risk
        {'range': [80, 100], 'color': "#E74C3C"}     # Red for high risk
    ],
    'threshold': {
        'line': {'color': "darkblue", 'width': 4},   # Darker line for the threshold marker
        'thickness': 0.8,
        'value': risk_score
    }
}

    ))

    # Convert the beautified gauge chart to HTML
    gauge_html = fig.to_html(full_html=False, include_plotlyjs='cdn')


    # Combine the gauge chart and classification reasoning inside one container
    classification_html = f"""
        <div class="gauge-container" style="text-align:center; margin-top: 20px;">
            {gauge_html}
            <div class="classification-reason" style="margin-top: 20px;">
                <h3 style="color: darkblue;">Classification Reasoning:</h3>
                <p style="color: #333; font-size: 20px; line-height: 1.6;">
                    {classification_reason.replace(f"{average_days_in_arrears:.2f}", f"<strong>{average_days_in_arrears:.2f}</strong>")
                                        .replace(f"{late_payer_percentage:.2f}%", f"<strong>{late_payer_percentage:.2f}%</strong>")
                    }
                </p>
            </div>
        </div>
        """


    # Payment trend over time
    customer_df['Due Date'] = pd.to_datetime(customer_df['Due Date'])
    payment_trend = customer_df.groupby(customer_df['Due Date'].dt.to_period('M'))['Predicted Days in Arrears'].mean().reset_index()
    payment_trend['Due Date'] = payment_trend['Due Date'].dt.to_timestamp()

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=payment_trend['Due Date'],
        y=payment_trend['Predicted Days in Arrears'],
        mode='lines+markers',
        name='Payment Trend'
    ))
    trend_fig.update_layout(
        title='Predicted Days in Arrears Over Time',
        xaxis_title='Duration',
        yaxis_title='Predicted Days in Arrears'
    )

    # Standard Amount in USD trend over time
    amount_trend = customer_df.groupby(customer_df['Due Date'].dt.to_period('M'))['Standard Amount in USD'].sum().reset_index()
    amount_trend['Due Date'] = amount_trend['Due Date'].dt.to_timestamp()

    amount_fig = go.Figure()
    amount_fig.add_trace(go.Scatter(
        x=amount_trend['Due Date'],
        y=amount_trend['Standard Amount in USD'],
        mode='lines+markers',
        name='Amount Trend'
    ))
    amount_fig.update_layout(
        title='Standard Amount in USD Over Time',
        xaxis_title='Duration',
        yaxis_title='Standard Amount in USD'
    )

    # Convert the figures to HTML
    gauge_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    trend_html = trend_fig.to_html(full_html=False, include_plotlyjs='cdn')
    amount_html = amount_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Return insights and charts via JSON
    return jsonify({
        'insights_html': insights_html + aging_html + aging_description,
        'gauge_data': fig.to_json(),
        'trend_data': trend_fig.to_json(),
        'amount_data': amount_fig.to_json(),
        'classification_html': classification_html
    })















































# Load the AR and AP datasets
file_path = 'AR_cashflow_forecasting_dataset.csv'
df_AR = pd.read_csv(file_path)

AP_file_path = 'AP_cashflow_payables_dataset.csv'
df_AP = pd.read_csv(AP_file_path)

# Preprocess the datasets
df_prophet_AR_aggregated = df_AR.groupby('Invoice Date').agg({'Amount Received': 'sum'}).reset_index()
df_prophet_AR_aggregated = df_prophet_AR_aggregated.rename(columns={'Invoice Date': 'ds', 'Amount Received': 'y'})
df_prophet_AR_aggregated['ds'] = pd.to_datetime(df_prophet_AR_aggregated['ds'])

df_prophet_AP_aggregated = df_AP.groupby('Invoice Date').agg({'Amount Payable': 'sum'}).reset_index()
df_prophet_AP_aggregated = df_prophet_AP_aggregated.rename(columns={'Invoice Date': 'ds', 'Amount Payable': 'y'})
df_prophet_AP_aggregated['ds'] = pd.to_datetime(df_prophet_AP_aggregated['ds'])

# Initialize Prophet models
model_AR = Prophet(changepoint_prior_scale=0.1)
model_AR.fit(df_prophet_AR_aggregated)

model_AP = Prophet(changepoint_prior_scale=0.1)
model_AP.fit(df_prophet_AP_aggregated)

# Function to forecast data
def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    future_forecast = forecast[forecast['ds'] > model.history['ds'].max()]
    return future_forecast

# Function to calculate KPIs
def calculate_kpis():
    total_inflow = df_prophet_AR_aggregated['y'].sum()
    total_outflow = df_prophet_AP_aggregated['y'].sum()
    net_cash_flow = total_inflow - total_outflow
    
    # Calculate days with surplus and deficit
    merged_df = pd.merge(df_prophet_AR_aggregated, df_prophet_AP_aggregated, on='ds', how='outer', suffixes=('_inflow', '_outflow')).fillna(0)
    merged_df['net_flow'] = merged_df['y_inflow'] - merged_df['y_outflow']
    days_with_surplus = len(merged_df[merged_df['net_flow'] > 0])
    days_with_deficit = len(merged_df[merged_df['net_flow'] < 0])

    kpis = {
        "total_inflow": total_inflow,
        "total_outflow": total_outflow,
        "net_cash_flow": net_cash_flow,
        "days_with_surplus": days_with_surplus,
        "days_with_deficit": days_with_deficit
    }
    return kpis

# Function to forecast yearly inflow/outflow for 2025
def forecast_yearly_cash_flow():
    future_AR = make_forecast(model_AR, 365)
    future_AP = make_forecast(model_AP, 365)

    future_AR_2025 = future_AR[future_AR['ds'].dt.year == 2025]
    future_AP_2025 = future_AP[future_AP['ds'].dt.year == 2025]

    inflow_2025 = future_AR_2025.resample('M', on='ds').sum(numeric_only=True).reset_index()
    outflow_2025 = future_AP_2025.resample('M', on='ds').sum(numeric_only=True).reset_index()

    cash_flow_fig = go.Figure()
    cash_flow_fig.add_trace(go.Bar(
        x=inflow_2025['ds'].dt.strftime('%B'),
        y=inflow_2025['yhat'],
        name='Inflow (Forecasted)',
        marker_color='blue'
    ))
    cash_flow_fig.add_trace(go.Bar(
        x=outflow_2025['ds'].dt.strftime('%B'),
        y=outflow_2025['yhat'],
        name='Outflow (Forecasted)',
        marker_color='red'
    ))

    cash_flow_fig.update_layout(
        title='Forecasted Cash Inflow and Outflow for 2025',
        xaxis_title='Month',
        yaxis_title='Amount',
        barmode='group',
        showlegend=True,
        template='plotly_white'
    )

    return cash_flow_fig

@app.route('/forecast.html')
def forecast():
    return render_template('forecast.html')

@app.route('/plots/<type>/<int:days>')
def plots(type, days):
    if type == 'ar':
        data = df_prophet_AR_aggregated
        model = model_AR
    elif type == 'ap':
        data = df_prophet_AP_aggregated
        model = model_AP
    else:
        return jsonify({"error": "Invalid type"}), 400

    future_forecast = make_forecast(model, days)
    
    forecast_fig = go.Figure()
    # Adding only the forecasted data
    forecast_fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat'],
        mode='lines',
        name='Forecasted Data',
        line=dict(color='orange', width=2, dash='dash', shape='spline', smoothing=1.2),
        marker=dict(color='orange', size=4, opacity=0.8)
    ))
    
    forecast_fig.update_layout(
        title={
            'text': f"Forecast for {days} Days - {type.upper()}",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='grey'
        ),
        yaxis=dict(
            title="Amount",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='grey'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white',
    )

    yearly_cash_flow_fig = forecast_yearly_cash_flow()

    # Get KPIs
    kpis = calculate_kpis()

    # Convert forecast data to a dictionary for the table
    forecast_table = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')

    data = {
        "forecast_fig": forecast_fig.to_json(),
        "yearly_cash_flow_fig": yearly_cash_flow_fig.to_json(),
        "forecast_table": forecast_table,
        "kpis": kpis
    }
    
    return jsonify(data)

















































def load_data():
    data = pd.read_csv('transaction_anomaly_dataset.csv', encoding='ISO-8859-1')
    # Keep the date columns as strings without converting to datetime
    data['invoice_date'] = data['invoice_date'].astype(str)
    data['due_date'] = data['due_date'].astype(str)
    data['payment_date'] = data['payment_date'].astype(str)
    
    # Ensure payment_difference is an integer
    data['payment_difference'] = (data['invoice_amount'] - data['payment_amount']).fillna(0).astype(int)
    
    return data

data = load_data()

@app.route('/anomaly.html')
def anomaly_page():
    customers = ['Overall'] + data['Customer Name'].dropna().unique().tolist()
    return render_template('anomaly.html', customers=customers)

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    try:
        # Get selected customer from the request
        selected_customer = request.json.get('customer')

        # Filter data based on selected customer for KPIs and table only
        if selected_customer != "Overall":
            filtered_data = data[data['Customer Name'] == selected_customer]
        else:
            filtered_data = data

        # Calculate KPIs based on filtered data
        total_amount_transacted = filtered_data['invoice_amount'].sum()
        total_anomalies = filtered_data['is_anomalous'].sum()
        average_amount_transacted = filtered_data['invoice_amount'].mean()
        average_risk_score = filtered_data['risk_score'].mean()
        average_days_past_due = filtered_data['days_past_due'].mean()
        anomaly_rate_percent = (total_anomalies / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0

        # Calculate the payment difference
        filtered_data['payment_difference'] = (filtered_data['invoice_amount'] - filtered_data['payment_amount']).fillna(0).astype(int)

        # **Filter to only include anomalies**
        anomaly_data = filtered_data[filtered_data['is_anomalous'] == 1]

        # Convert the anomaly data to JSON for the table
        table_data = anomaly_data[['invoice_id', 'invoice_date', 'due_date', 'invoice_amount', 'Country', 'Customer Name', 
                                   'payment_date', 'payment_amount', 'days_past_due', 'payment_difference', 'is_anomalous','anomaly_type']].to_dict(orient='records')

        # Generate static Plotly figures based on the overall dataset (unchanged charts)
        plots = generate_static_plots(data)

        return jsonify({
            'status': 'success',
            'kpis': {
                'total_amount_transacted': f"${total_amount_transacted:,.2f}",
                'total_anomalies': int(total_anomalies),
                'average_amount_transacted': f"${average_amount_transacted:,.2f}",
                'average_risk_score': f"{average_risk_score:.2f}",
                'average_days_past_due': f"{average_days_past_due:.2f} days",
                'anomaly_rate_percent': f"{anomaly_rate_percent:.2f}%"
            },
            'plots': plots,
            'table_data': table_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def generate_static_plots(full_data):
    plots = {}

    # Isolation Forest Anomaly Detection based on overall dataset
    features = ['invoice_amount', 'days_past_due']
    data_clean = full_data[features].dropna()
    isolation_forest = IsolationForest(contamination=0.01, random_state=42)
    data_clean['anomaly_score'] = isolation_forest.fit_predict(data_clean)
    data_clean['anomaly_score'] = np.where(data_clean['anomaly_score'] == -1, 1, 0)
    full_data['anomaly_score'] = 0
    full_data.loc[data_clean.index, 'anomaly_score'] = data_clean['anomaly_score']
    fig_isolation = px.scatter(full_data, x='days_past_due', y='invoice_amount', color='anomaly_score',
                               color_continuous_scale=['blue', 'red'], title="Isolation Forest Anomaly Detection",
                               labels={'payment_delay': 'Payment Delay (days)', 'invoice_amount': 'Invoice Amount ($)'},
                               hover_data=['Customer Name', 'Country', 'risk_score'])
    fig_isolation.update_layout(xaxis=dict(tickfont=dict(size=10)))
    plots['isolation_forest'] = pio.to_json(fig_isolation)

    # Geographical Distribution of Anomalies
    anomaly_map_data = full_data[full_data['is_anomalous'] == 1].groupby('Country').size().reset_index(name='anomalies')
    fig_map = px.choropleth(anomaly_map_data, locations='Country', locationmode='country names', color='anomalies',
                            title="Geographical Distribution of Anomalies", color_continuous_scale='Reds',
                            labels={'anomalies': 'Number of Anomalies'})
    plots['geographical_anomalies'] = pio.to_json(fig_map)

    # Anomalies by Type
    anomalies_by_type = full_data[full_data['is_anomalous'] == 1].groupby('anomaly_type')['invoice_id'].count().reset_index()
    fig_anomaly_type = px.bar(anomalies_by_type, x='anomaly_type', y='invoice_id', title="Anomalies by Type", labels={'invoice_id': 'Count of Invoices'}).update_layout(xaxis=dict(tickfont=dict(size=10)))
    plots['anomalies_by_type'] = pio.to_json(fig_anomaly_type)

    # Anomalies Distribution by Country (Top 5)
    anomaly_distribution = full_data[full_data['is_anomalous'] == 1].groupby('Country').size().reset_index(name='anomalies')
    top_anomaly_countries = anomaly_distribution.sort_values(by='anomalies', ascending=False).head(5)
    fig_pie = px.pie(top_anomaly_countries, values='anomalies', names='Country', title="Anomalies Distribution by Country (Top 5)")
    plots['anomalies_distribution_country'] = pio.to_json(fig_pie)

    return plots










































# Main Route for Insights Dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    suppliers = df['Supplier or Party'].unique()

    if request.method == 'POST':
        selected_supplier = request.form.get('supplier')
        highlight_option = request.form.get('highlight_option', 'Show All')
    else:
        selected_supplier = suppliers[0]
        highlight_option = 'Show All'

    supplier_data = df[df['Supplier or Party'] == selected_supplier]

    # Generate summary cards data
    cards = generate_summary_cards(supplier_data)

    # Apply conditional formatting to the dataframe
    styled_df = apply_highlight(supplier_data, highlight_option)

    # Generate all the plots
    plots = generate_all_plots(supplier_data)

    return render_template(
        'dashboard.html',
        suppliers=suppliers,
        selected_supplier=selected_supplier,
        cards=cards,
        highlight_option=highlight_option,
        supplier_data=styled_df.to_html(classes='table table-striped table-hover', index=False, escape=False),
        plots=plots
    )


def generate_summary_cards(supplier_data):
    cards = []
    total_invoices = len(supplier_data)
    cards.append(("Total Invoices", total_invoices))

    if 'Paid Status' in supplier_data.columns and 'Invoice Number' in supplier_data.columns:
        unpaid_invoices = supplier_data[supplier_data['Paid Status'] == 'Unpaid']
        total_unpaid = unpaid_invoices['Invoice Number'].count()
    else:
        total_unpaid = 0
    cards.append(("Unpaid Invoices", total_unpaid))

    if 'Time to Pay' in supplier_data.columns:
        avg_time_to_pay = supplier_data['Time to Pay'].mean()
        avg_time_to_pay = round(avg_time_to_pay) if not supplier_data['Time to Pay'].isna().all() else 0
    else:
        avg_time_to_pay = 0
    cards.append(("Average Time to Pay", f"{avg_time_to_pay} days"))


    if 'Adherence Difference' in supplier_data.columns:
        adherence_diff = supplier_data['Adherence Difference'].mean()
        adherence_diff = round(adherence_diff) if not supplier_data['Adherence Difference'].isna().all() else 0
    else:
        adherence_diff = 0
    cards.append(("Adherence Difference", f"{adherence_diff} days"))


    if 'Paid on Time' in supplier_data.columns:
        on_time_paid_count = supplier_data[supplier_data['Paid on Time metrics'] == 'Paid On Time']['Invoice Number'].count()
    else:
        on_time_paid_count = 0
    cards.append(("Paid on Time Invoices", on_time_paid_count))

    if 'Receipt date to Due date' in supplier_data.columns:
        avg_receipt_to_due = supplier_data['Receipt date to Due date'].mean()
        avg_receipt_to_due = round(avg_receipt_to_due) if not supplier_data['Receipt date to Due date'].isna().all() else 0
    else:
        avg_receipt_to_due = 0
    cards.append(("Avg Receipt to Due Date", f"{avg_receipt_to_due} days"))


    if 'creation to initiation for approval' in supplier_data.columns:
        avg_creation_to_initiation = supplier_data['creation to initiation for approval'].mean()
        avg_creation_to_initiation = round(avg_creation_to_initiation) if not supplier_data['creation to initiation for approval'].isna().all() else 0
    else:
        avg_creation_to_initiation = 0
    cards.append(("Avg Creation to Initiation", f"{avg_creation_to_initiation} days"))

    if 'Initiation for approval to Payment' in supplier_data.columns:
        avg_initiation_to_approval = supplier_data['Initiation for approval to Payment'].mean()
        avg_initiation_to_approval = round(avg_initiation_to_approval) if not supplier_data['Initiation for approval to Payment'].isna().all() else 0
    else:
        avg_initiation_to_approval = 0
    cards.append(("Avg Initiation to Approval", f"{avg_initiation_to_approval} days"))


    total_invoice_amount = supplier_data['Invoice Amount'].sum()
    avg_invoice_amount = supplier_data['Invoice Amount'].mean()
    cards.append(("Total Invoice Amount", f"${total_invoice_amount:,.2f}"))
    cards.append(("Average Invoice Amount", f"${avg_invoice_amount:,.2f}"))

    supplier_site = supplier_data['Supplier Site'].unique()[0] if 'Supplier Site' in supplier_data.columns and not supplier_data['Supplier Site'].isna().all() else "N/A"
    cards.append(("Supplier Site", supplier_site))

    invoice_type = supplier_data['Invoice Type'].unique()[0] if 'Invoice Type' in supplier_data.columns and not supplier_data['Invoice Type'].isna().all() else "N/A"
    cards.append(("Invoice Type", invoice_type))

    if 'Validation Status' in supplier_data.columns:
        validated_count = supplier_data[supplier_data['Validation Status'] == 'VALIDATED']['Invoice Number'].count()
        validated_percent = (validated_count / total_invoices) * 100 if total_invoices > 0 else 0
    else:
        validated_percent = 0
    cards.append(("Validation %", f"{validated_percent:.2f}%"))

    if 'Accounting Status' in supplier_data.columns:
        accounted_count = supplier_data[supplier_data['Accounting Status'] == 'Accounted']['Invoice Number'].count()
        unaccounted_count = supplier_data[supplier_data['Accounting Status'] == 'Unaccounted']['Invoice Number'].count()
    else:
        accounted_count = unaccounted_count = 0
    cards.append(("Accounting Status", f"Accounted: {accounted_count} | Unaccounted: {unaccounted_count}"))

    pay_group = supplier_data['Pay Group'].unique()[0] if 'Pay Group' in supplier_data.columns and not supplier_data['Pay Group'].isna().all() else "N/A"
    payment_method = supplier_data['Payment_Method'].unique()[0] if 'Payment_Method' in supplier_data.columns and not supplier_data['Payment_Method'].isna().all() else "N/A"
    cards.append(("Pay Group", pay_group))
    cards.append(("Payment Method", payment_method))


    return cards

def apply_highlight(supplier_data, highlight_option):
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in supplier_data.columns:
        supplier_data = supplier_data.drop(columns=['Unnamed: 0'])

    def highlight_status(row):
        if highlight_option == 'Show All':
            if 'Validation Status' in row and str(row['Validation Status']).strip().lower() == 'cancelled':
                return ['background-color: yellow'] * len(row)
            elif 'Paid on Time metrics' in row and str(row['Paid on Time metrics']).strip().lower() == 'paid late':
                return ['background-color: lightcoral'] * len(row)
            elif 'Received onTime' in row and str(row['Received onTime']).strip().lower() == 'received late':
                return ['background-color: red'] * len(row)
            else:
                return [''] * len(row)
        else:
            if highlight_option == 'Cancelled' and 'Validation Status' in row and str(row['Validation Status']).strip().lower() == 'cancelled':
                return ['background-color: yellow'] * len(row)
            elif highlight_option == 'Paid Late' and 'Paid on Time metrics' in row and str(row['Paid on Time metrics']).strip().lower() == 'paid late':
                return ['background-color: lightcoral'] * len(row)
            elif highlight_option == 'Received Late' and 'Received onTime' in row and str(row['Received onTime']).strip().lower() == 'received late':
                return ['background-color: red'] * len(row)
            else:
                return [''] * len(row)

    # Apply the highlight and set table attributes for horizontal scrolling
    styled_df = supplier_data.style.apply(highlight_status, axis=1).set_table_attributes(
        'style="overflow-x: auto; display: block; width: 100%;"'
    )
    return styled_df


def generate_all_plots(supplier_data):
    plots = []

    # Example 1: Bar chart for Invoice Amounts by PO Status
    if 'PO Status' in supplier_data.columns and 'Invoice Amount' in supplier_data.columns:
        po_status_amount = supplier_data.groupby('PO Status')['Invoice Amount'].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        po_status_amount.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        set_professional_style(ax, "Total Invoice Amount by PO Status", "PO Status", "Total Invoice Amount")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 2: Line chart for Payment Delays
    if 'Day Difference' in supplier_data.columns and 'Due Month' in supplier_data.columns:
        avg_day_diff = supplier_data.groupby('Due Month')['Day Difference'].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_day_diff.plot(kind='line', ax=ax, marker='o', color='purple', linestyle='-', linewidth=2, markersize=6)
        set_professional_style(ax, "Average Payment Delay by Due Month", "Due Month", "Average Delay (Days)")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 3: Histogram of Invoice Amounts
    if 'Invoice Amount' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(supplier_data['Invoice Amount'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        set_professional_style(ax, "Distribution of Invoice Amounts", "Invoice Amount", "Frequency")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 4: Scatter plot for Invoice Amount vs Time to Pay
    if 'Invoice Amount' in supplier_data.columns and 'Time to Pay' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(supplier_data['Time to Pay'], supplier_data['Invoice Amount'], color='darkred', alpha=0.6, edgecolors='black')
        set_professional_style(ax, "Invoice Amount vs. Time to Pay", "Time to Pay (Days)", "Invoice Amount")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 5: Time to Pay vs Approval Status
    if 'Time to Pay' in supplier_data.columns and 'Approval Status' in supplier_data.columns:
        approval_groups = supplier_data.groupby('Approval Status')['Time to Pay'].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        approval_groups.plot(kind='bar', ax=ax, color='darkorange', edgecolor='black')
        set_professional_style(ax, "Average Time to Pay by Approval Status", "Approval Status", "Average Time to Pay (Days)")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 6: Payment Terms Analysis
    if 'Payment Term Days' in supplier_data.columns and 'Time to Pay' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(supplier_data['Payment Term Days'], supplier_data['Time to Pay'], color='blue', alpha=0.7, edgecolors='black')
        set_professional_style(ax, "Payment Term Days vs Time to Pay", "Payment Term Days", "Time to Pay (Days)")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 7: Time to Pay vs Invoice Amount by Supplier Site
    if 'Time to Pay' in supplier_data.columns and 'Invoice Amount' in supplier_data.columns and 'Supplier Site' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        supplier_data.groupby('Supplier Site').plot.scatter(x='Time to Pay', y='Invoice Amount', ax=ax, alpha=0.6)
        set_professional_style(ax, "Time to Pay vs Invoice Amount by Supplier Site", "Time to Pay (Days)", "Invoice Amount")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 8: Paid on Time vs Paid Late (Pie Chart)
    if 'Paid on Time metrics' in supplier_data.columns:
        paid_on_time = supplier_data['Paid on Time metrics'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(paid_on_time, labels=paid_on_time.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90, textprops={'fontsize': 12, 'color': 'black'}, wedgeprops={'edgecolor': 'black'})
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_color('white')
            autotext.set_weight('bold')
        for text in texts:
            text.set_fontsize(12)
            text.set_color('gray')
        ax.axis('equal')
        plt.title('Paid on Time vs Paid Late', fontsize=14, fontweight='bold', color='darkblue', pad=15)
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 9: Bar chart for Day Difference vs. PO Status
    if 'Day Difference' in supplier_data.columns and 'PO Status' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        po_status_diff = supplier_data.groupby('PO Status')['Day Difference'].mean()
        po_status_diff.plot(kind='bar', yerr=supplier_data.groupby('PO Status')['Day Difference'].std(), ax=ax, color='teal', edgecolor='black')
        set_professional_style(ax, "Day Difference by PO Status", "PO Status", "Day Difference (Days)")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())



    # Example 11: Creation to Initiation for Approval vs Initiation to Payment (Scatter Plot)
    if 'creation to initiation for approval' in supplier_data.columns and 'Initiation for approval to Payment' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(supplier_data['creation to initiation for approval'], supplier_data['Initiation for approval to Payment'], color='darkred', alpha=0.6, edgecolors='black')
        set_professional_style(ax, "Creation to Initiation vs Initiation to Payment", "Creation to Initiation for Approval (Days)", "Initiation for Approval to Payment (Days)")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 12: Bar chart for Invoice to Receipt and Receipt Date to Due Date
    if 'Invoice to Receipt' in supplier_data.columns and 'Receipt date to Due date' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Invoice to Receipt', 'Receipt Date to Due Date'], [supplier_data['Invoice to Receipt'].mean(), supplier_data['Receipt date to Due date'].mean()], color=['blue', 'green'], edgecolor='black')
        set_professional_style(ax, "Average Invoice to Receipt and Receipt Date to Due Date", "Metrics", "Average Days")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 13: Validation Status (Bar Chart)
    if 'Validation Status' in supplier_data.columns:
        validation_status = supplier_data['Validation Status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        validation_status.plot(kind='bar', ax=ax, color=['green', 'red'], edgecolor='black')
        set_professional_style(ax, "Validation Status", "Status", "Count")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 14: Received On Time vs Received Late (Pie Chart)
    if 'Received onTime' in supplier_data.columns:
        received_on_time_data = supplier_data['Received onTime'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(received_on_time_data, labels=received_on_time_data.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90, textprops={'fontsize': 12, 'color': 'black'}, wedgeprops={'edgecolor': 'black'})
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_color('white')
            autotext.set_weight('bold')
        for text in texts:
            text.set_fontsize(12)
            text.set_color('gray')
        ax.axis('equal')
        plt.title('Received On Time vs Received Late', fontsize=14, fontweight='bold', color='darkblue', pad=15)
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 15: Day Difference vs. Payment Terms and Invoice Type (Bar Chart)
    if 'Day Difference' in supplier_data.columns and 'Payment Terms' in supplier_data.columns and 'Invoice Type' in supplier_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        day_diff_payment_invoice = supplier_data.groupby(['Payment Terms', 'Invoice Type'])['Day Difference'].mean().unstack()
        day_diff_payment_invoice.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c'], edgecolor='black')
        set_professional_style(ax, "Day Difference by Payment Terms and Invoice Type", "Payment Terms", "Day Difference (Days)")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    # Example 17: Payment Month and Received Late (Stacked Bar Chart)
    if 'Payment Month' in supplier_data.columns and 'Received Late Month' in supplier_data.columns:
        received_late = supplier_data.groupby(['Payment Month', 'Received Late Month']).size().unstack()
        fig, ax = plt.subplots(figsize=(6, 4))
        received_late.plot(kind='bar', stacked=True, ax=ax, color=['#ff7f0e', '#2ca02c'], edgecolor='black')
        set_professional_style(ax, "Payment Month vs Received Late", "Payment Month", "Number of Late Receipts")
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plots.append(base64.b64encode(img.getvalue()).decode())

    return plots

if __name__ == '__main__':
    app.run(debug=True)


