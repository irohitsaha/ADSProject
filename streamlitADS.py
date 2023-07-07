import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("Automatic Data Analyser")

upload_file = st.file_uploader("Upload Your CSV File For Data Analyse", type="csv")

if upload_file is not None:
        st.write("**File Upload Successfully**")
        st.subheader("The Uploaded DataSet is : ")
        df = pd.read_csv(upload_file)
        st.dataframe(df)

        st.subheader("Description of Your DataSet : ")
        st.write(df.describe())
        st.write("Handling Null Values:")
        null_counts = df.isnull().sum()
        st.write("Null Value Counts:")
        st.write(null_counts)

        df.dropna(inplace=True)

        st.write("**Data Visualization**")

        st.subheader("Bar Chart")
        column = st.selectbox("Select a column for the Bar Chart", df.columns)
        st.bar_chart(df[column].value_counts())

        st.subheader("Line Chart")
        numeric_column = df.select_dtypes(include=[int, float]).columns
        column = st.selectbox("Select a column for the Line Chart",numeric_column)
        st.line_chart(df[column])

        st.subheader("Histogram")
        column = st.selectbox("Select a column for the Histogram",numeric_column)
        fig, ax = plt.subplots()
        ax.hist(df[column])
        st.pyplot(fig)

        st.subheader("Scatter Plot")
        x_axis = st.selectbox("select x axis for the scatter plot",numeric_column)
        y_axis = st.selectbox("select y axis for the scatter plot",numeric_column)
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis])
        st.pyplot(fig)

        st.subheader("**Logistic Regression**")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        st.write("Dataset is: ")
        st.dataframe(df)

        y_pred = model.predict(X_test)
        accuracy_score = model.score(X_test, y_test)
        st.write("Model Accuracy Score:", accuracy_score)

        report = classification_report(y_test, y_pred)
        st.write("Classification Report:")
        st.text(report)

        st.write("Please enter the values for prediction:")
        prediction_inputs = []
        for column in X.columns:
            value = st.number_input(column)
            prediction_inputs.append(value)
        
        prediction = model.predict([prediction_inputs])

        st.write("Prediction:")
        st.write(prediction)

#Done by Rohit











    
