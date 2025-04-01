import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/ErenKozak/Churn-Prediction/refs/heads/master/Data/customer_churn_dataset-testing-master.csv")

# Convert categorical variables to numerical values
label_encoders = {}
for col in ['Gender', 'Subscription Type', 'Contract Length']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare the dataset
X = df.drop(columns=["Churn", "CustomerID"])  # Independent variables
y = df['Churn']  # Dependent variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit UI
st.title('🏢 Churn Prediction')

st.write('Enter your data below to predict customer churn!')

with st.expander("**Data**"):
    st.dataframe(df)
    
    column_definitions = {
        'Age': 'Customer age.',
        'Gender': 'Customer gender.',
        'Tenure': 'Time spent with the service provider.',
        'Usage Frequency': 'Frequency of service usage.',
        'Support Calls': 'Number of support calls made by the customer.',
        'Payment Delay': 'Payment delay duration.',
        'Subscription Type': 'Type of subscription.',
        'Contract Length': 'Contract duration.',
        'Total Spend': 'Total amount spent.',
        'Last Interaction': 'Time since last customer interaction.'
    }

    st.title('Column Definitions')
    for column, definition in column_definitions.items():
        st.write(f"**{column}**: {definition}")

st.title('Customer Churn Analysis')
churn_data = df['Churn'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=churn_data.index, y=churn_data.values, palette="pastel", ax=ax)
ax.set_title('Customer Churn Rate')
ax.set_xlabel('Churn Status')
ax.set_ylabel('Number of Customers')
st.pyplot(fig)

with st.expander("**Enter Data**"):
    age = st.slider("Select Your Age", min_value=18, max_value=100, value=31)
    gender = st.selectbox("Select Your Gender", ("Female", "Male"))
    tenure = st.slider("Select Tenure", min_value=1, max_value=60, value=31)
    usage_frequency = st.slider("Usage Frequency", min_value=1, max_value=30, value=14)
    support_calls = st.slider("Support Calls", min_value=0, max_value=10, value=4)
    payment_delay = st.slider("Payment Delay", min_value=0, max_value=30, value=14)
    subscription_type = st.selectbox("Select Subscription Type", ("Basic", "Standard", "Premium"))
    contract_length = st.selectbox("Select Contract Length", ("Monthly", "Annual", "Quarterly"))
    total_spend = st.slider("Total Spend", min_value=100, max_value=1000, value=450)
    last_interaction = st.slider("Last Interaction", min_value=1, max_value=30, value=14)

# Convert user inputs into a NumPy array
input_data = np.array([[age, gender, tenure, usage_frequency, support_calls, 
                        payment_delay, subscription_type, contract_length, 
                        total_spend, last_interaction]], dtype=object)

# Convert categorical inputs
if input_data[0, 1] == "Female":
    input_data[0, 1] = 0
elif input_data[0, 1] == "Male":
    input_data[0, 1] = 1

if input_data[0, 6] == "Basic":
    input_data[0, 6] = 0
elif input_data[0, 6] == "Standard":
    input_data[0, 6] = 2
elif input_data[0, 6] == "Premium":
    input_data[0, 6] = 1

if input_data[0, 7] == "Monthly":
    input_data[0, 7] = 1
elif input_data[0, 7] == "Annual":
    input_data[0, 7] = 0
elif input_data[0, 7] == "Quarterly":
    input_data[0, 7] = 2

# Convert numeric values
input_data[0, 0] = float(input_data[0, 0])  # age
input_data[0, 2] = int(input_data[0, 2])  # tenure
input_data[0, 3] = int(input_data[0, 3])  # usage_frequency
input_data[0, 4] = int(input_data[0, 4])  # support_calls
input_data[0, 5] = int(input_data[0, 5])  # payment_delay
input_data[0, 7] = int(input_data[0, 7])  # contract_length
input_data[0, 9] = float(input_data[0, 9])  # total_spend

# Convert NumPy array to float
data_input = input_data.astype(float)

# Model prediction
prediction = rf_model.predict(data_input)

# Display prediction result
if prediction == [1]:
    st.markdown(
    """
    <div style="background-color: rgba(240, 59, 53, 0.7); padding: 10px; border-radius: 5px;">
        <p style="color: white; text-align: center; font-size: 20px;">Prediction: The customer will churn.</p>
    </div>
    """,
    unsafe_allow_html=True
)
else:
    st.markdown(
    """
    <div style="background-color: rgba(127, 237, 76, 0.7); padding: 10px; border-radius: 5px;">
        <p style="color: white; text-align: center; font-size: 20px;">Prediction: The customer will not churn.</p>
    </div>
    """,
    unsafe_allow_html=True
)
    
