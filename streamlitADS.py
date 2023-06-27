import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


st.title("Automatic Data Analyser")

upload_file = st.file_uploader("Upload Your CSV File For Data Analyse", type="csv")

if upload_file is not None:
        st.write("**File Upload Successfully**")
        st.subheader("The Uploaded DataSet is : ")
        df = pd.read_csv(upload_file)
        st.dataframe(df)

        st.subheader("Description of Your DataSet : ")
        st.write(df.describe())

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

#Done by Rohit











    
