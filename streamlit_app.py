import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


df = pd.read_csv("https://raw.githubusercontent.com/ErenKozak/Churn-Prediction/refs/heads/master/Data/customer_churn_dataset-testing-master.csv")

# KATEGORİK DEĞİŞKENLERİ SAYISALLAŞTIRMA
label_encoders = {}
for col in ['Gender', 'Subscription Type', 'Contract Length']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# VERİ SETİNİ HAZIRLAMA
X = df.drop(columns=["Churn","CustomerID"])  # Bağımsız değişkenler
y = df['Churn']  # Bağımlı değişken

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. MODEL EĞİTME (RANDOM FOREST)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

st.title('🏢 Churn Predict')

st.write('Aşağıdaki alanlara kendi verilerinizi girerek Churn tahmininizi yapabilirsiniz!')


with st.expander("**Data**"):
    st.dataframe(df)

    column_definitions = {
    'Age': 'Müşterinin yaşı.',
    'Gender': 'Müşterinin cinsiyeti.',
    'Tenure': 'Müşterinin hizmet sağlayıcıyla geçirdiği süre.',
    'Usage Frequency': 'Müşterinin hizmeti ne sıklıkta kullandığı.',
    'Support Calls': 'Müşterinin destek için yaptığı çağrı sayısı.',
    'Payment Delay': 'Müşterinin ödeme gecikme süresi.',
    'Subscription Type': 'Müşterinin abone olduğu hizmet tipi.',
    'Contract Length': 'Müşterinin sözleşme süresi.',
    'Total Spend': 'Müşterinin toplam harcaması.',
    'Last Interaction': 'Müşteri ile yapılan son etkileşimin tarihi veya süresi.'
}

    # Başlık
    st.title('Kolon Tanımlamaları')

    # Tanımlamaları metin olarak göster
    for column, definition in column_definitions.items():
        st.write(f"**{column}**: {definition}")


st.title('Müşteri Churn Analizi')
churn_data = df['Churn'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=churn_data.index, y=churn_data.values,palette="pastel",ax=ax)
ax.set_title('Müşterilerin hizmetinizi kullanmayı bırakma oranı')
ax.set_xlabel('Churn Durumu')
ax.set_ylabel('Müşteri Sayısı')
st.pyplot(fig)

with st.expander("**Veri Girişi**"):
    age = st.slider("Yaşınızı Seçiniz", min_value=18, max_value=100, value=31)
    gender = st.selectbox("Cinsiyetinizi seçin", ("Kadın", "Erkek"))
    tenure = st.slider("Tenure Seçiniz", min_value=1, max_value=60, value=31)
    usage_frequency = st.slider("Usage Frequency", min_value=1, max_value=30, value=14)
    support_calls = st.slider("Support Calls", min_value=0, max_value=10, value=4)
    payment_delay = st.slider("Payment Delay", min_value=0, max_value=30, value=14)
    subscription_type = st.selectbox("Subscription Type seçin", ("Basic", "Standard","Premium"))
    contract_length = st.selectbox("Contract Length seçin", ("Monthly", "Annual","Quarterly"))
    total_spend = st.slider("Total Spend", min_value=100, max_value=1000, value=450)
    last_interaction = st.slider("Last Interaction", min_value=1, max_value=30, value=14)

# Kullanıcıdan gelen verileri içeren numpy dizisi
input_data = np.array([[age, gender, tenure, usage_frequency, support_calls, 
                        payment_delay, subscription_type, contract_length, 
                        total_spend, last_interaction]], dtype=object)

# Cinsiyet dönüşümü
if input_data[0, 1] == "Kadın":
    input_data[0, 1] = 0
elif input_data[0, 1] == "Erkek":
    input_data[0, 1] = 1

# Abonelik tipi dönüşümü
if input_data[0, 6] == "Basic":
    input_data[0, 6] = 0
elif input_data[0, 6] == "Standard":
    input_data[0, 6] = 2
elif input_data[0, 6] == "Premium":
    input_data[0, 6] = 1

# Kontrat uzunluğu dönüşümü
if input_data[0, 7] == "Monthly":  # Doğru index: 7
    input_data[0, 7] = 1
elif input_data[0, 7] == "Annual":
    input_data[0, 7] = 0
elif input_data[0, 7] == "Quarterly":
    input_data[0, 7] = 2

# Sayısal dönüşümler
input_data[0, 0] = float(input_data[0, 0])  # age
input_data[0, 2] = int(input_data[0, 2])  # tenure
input_data[0, 3] = int(input_data[0, 3])  # usage_frequency
input_data[0, 4] = int(input_data[0, 4])  # support_calls
input_data[0, 5] = int(input_data[0, 5])  # payment_delay
input_data[0, 7] = int(input_data[0, 7])  # contract_length
input_data[0, 9] = float(input_data[0, 9])  # total_spend

# NumPy array'ini float türüne çevir
input_data = input_data.astype(float)

# Model tahmini
prediction = rf_model.predict(input_data)

# Tahmini yazdır
if prediction == [1]:
    st.markdown(
    """
    <div style="background-color: rgba(240, 59, 53, 0.7); padding: 10px; border-radius: 5px;">
        <p style="color: white; text-align: center; font-size: 20px;">Tahmin: Müşteri ürünü terk edecek.</p>
    </div>
    """,
    unsafe_allow_html=True
)
else:
    st.markdown(
    """
    <div style="background-color: rgba(127, 237, 76, 0.7); padding: 10px; border-radius: 5px;">
        <p style="color: white; text-align: center; font-size: 20px;">Tahmin: Müşteri ürünü terk etmeyecek.</p>
    </div>
    """,
    unsafe_allow_html=True
)
