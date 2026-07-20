import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

st.title("🏠 House Price Prediction App with Graphs")

# Upload dataset
uploaded_file = st.file_uploader("Upload your house prices CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Show correlation heatmap
    st.write("### Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Features and target
    features = st.multiselect("Select features", data.columns.tolist(), default=['area','bedrooms','bathrooms','age'])
    target = st.selectbox("Select target column", data.columns.tolist(), index=data.columns.tolist().index('price'))

    # Linear Regression
    if st.button("Run Linear Regression"):
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.write("#### Linear Regression Results")
        st.write("RMSE:", rmse)
        st.write("R²:", r2_score(y_test, y_pred))

        # Scatter plot of actual vs predicted
        st.write("### Actual vs Predicted Prices")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        st.pyplot(fig)

    # Logistic Regression
    if st.button("Run Logistic Regression"):
        threshold = st.number_input("Set price threshold", value=5000000)
        data['price_category'] = (data[target] > threshold).astype(int)

        X = data[features]
        y = data['price_category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)
        y_pred = log_model.predict(X_test)

        st.write("#### Logistic Regression Results")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))

        # Bar chart of class distribution
        st.write("### Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='price_category', data=data, ax=ax)
        ax.set_xticklabels(["Affordable (0)", "Expensive (1)"])
        st.pyplot(fig)

        # Confusion matrix heatmap
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Affordable","Expensive"], yticklabels=["Affordable","Expensive"], ax=ax)
        st.pyplot(fig)
