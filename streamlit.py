import streamlit as st
import joblib
import numpy as np
from PIL import Image

from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


import plotly.express as px 

from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  

# File Handling Libraries
import joblib  # For loading and saving model and scaler files
from io import StringIO  # For capturing output (e.g., info() from pandas)

# Load the scaler and prediction model from pickle files
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="💉",
    layout="centered",
)

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Diabetes Prediction Model", "About the Model", "EDA of Original Data Set"])

theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown('<style>body { background-color: #2B2B2B; color: white; }</style>', unsafe_allow_html=True)

# Page 1: Diabetes Prediction Model
if page == "Diabetes Prediction Model":
    # Header with custom styling
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #4CAF50;
            font-weight: bold;
        }
        .description {
            text-align: center;
            font-size: 1.2rem;
            color: #555555;
        }
        </style>
        <div class="main-title">Diabetes Prediction Model</div>
        <div class="description">Predict diabetes risk using machine learning.</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Input fields for prediction
    st.write("### Enter the following details:")
    
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, format="%d")
    glucose = st.slider("Glucose Level", 0, 250, 120)
    blood_pressure = st.slider("Blood Pressure in mm Hg", 0, 150, 80)
    skin_thickness = st.slider("Skin Thickness in mm", 0, 80, 20)
    insulin = st.slider("Insulin Level in muU/ml", 0, 150, 50)
    bmi = st.slider("BMI as (weight in kg/height in m)^2", 0.0, 80.0, 25.0, step=0.1)
    diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
    age = st.number_input("Age", min_value=0, step=1, format="%d")
    
    # Predict button
    if st.button("Predict"):
        # Prepare the input features as an array
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Display result with visual feedback
        if prediction[0] == 1:
            st.error("The patient is likely to have diabetes.")
            st.image("sad_smiley.jpg", caption="Take care of your health!", use_column_width=True)
            st.markdown(
                """
                ### Recommendations:
                - Consult a healthcare provider immediately.
                - Maintain a balanced diet and exercise regularly.
                - Monitor glucose levels frequently.
                - Avoid processed and sugary foods.
                """
            )


        else:
            st.success("The patient is not likely to have diabetes.")
            st.image("smiling_smiley.jpg", caption="Stay healthy!", use_column_width=True)
            st.markdown(
                """
                ### Recommendations:
                - Keep following a healthy lifestyle.
                - Regularly check your health with routine medical check-ups.
                - Stay active and eat a balanced diet.
                """
            )

            

        if prediction[0] == 0:
            st.balloons()
            st.markdown("🎉 Congratulations! Keep up the healthy lifestyle!")


    
        


    if st.button("Predict with Probability"):
        # Prepare the input features as an array
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        
        # Scale the features
        scaled_features = scaler.transform(features)

        probabilities = model.predict_proba(scaled_features)[0]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilities[1],  # Probability of diabetes
            title={'text': "Diabetes Probability"},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "red"}},
        ))
        st.plotly_chart(gauge)

    with st.expander("See Tips for Preventing Diabetes"):
        st.markdown("""
        - Regular exercise and a balanced diet can help lower your risk.
        - Include more vegetables and whole grains in your meals.
        - Avoid processed and sugary foods.
        """)


   

    




# Page 2: About the Model
elif page == "About the Model":
    st.title("About the Model")

    # Machine Learning Model Info
    st.header("Model Used: Random Forest Classifier")
    st.write("""
        The Random Forest Classifier is a versatile and robust machine learning model that creates multiple decision 
        trees during training and outputs the mode of the classes (classification) or mean prediction (regression) 
        of the individual trees. This model is trained with the following parameters:
    """)
    st.code("""
    from sklearn.ensemble import RandomForestClassifier
    RandomForest = RandomForestClassifier(n_estimators=150)
    RandomForest.fit(X_train_scaled_df, y_train)
    """)

    # Display Classification Report
    with st.expander("Classification Report"):
        classification_report_data = {
            "Metric": ["Precision", "Recall", "F1-Score", "Support"],
            "Class 0": [0.75, 0.87, 0.80, 142],
            "Class 1": [0.72, 0.54, 0.62, 89],
            "Accuracy": [0.74, "-", "-", 231],
            "Macro Avg": [0.73, 0.70, 0.71, 231],
            "Weighted Avg": [0.74, 0.74, 0.73, 231]
        }
        classification_report_df = pd.DataFrame(classification_report_data)
        st.dataframe(classification_report_df)

        # Confusion Matrices
    with st.expander("Confusion Matrices"):

        # Train Confusion Matrix
        st.write("Confusion Matrix for Train Set")
        train_cm = [[0, 179], [358, 0]]  # Values from screenshot
        train_fig = go.Figure(data=go.Heatmap(
            z=train_cm,
            x=["Predicted 0", "Predicted 1"],
            y=["True 0", "True 1"],
            colorscale="Viridis",
            text=train_cm,
            texttemplate="%{text}",
            showscale=True
        ))
        train_fig.update_layout(
            title="Train Set Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            font=dict(size=14)
        )
        st.plotly_chart(train_fig)

        # Test Confusion Matrix
        st.write("Confusion Matrix for Test Set")
        test_cm = [[123, 19], [48, 41]]  # Values from screenshot
        test_fig = go.Figure(data=go.Heatmap(
            z=test_cm,
            x=["Predicted 0", "Predicted 1"],
            y=["True 0", "True 1"],
            colorscale="Viridis",
            text=test_cm,
            texttemplate="%{text}",
            showscale=True
        ))
        test_fig.update_layout(
            title="Test Set Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            font=dict(size=14)
        )
        st.plotly_chart(test_fig)

    # Additional Insights
    with st.expander("Model Insights"):
   

        # Classification Report Visualization
        st.write("Visualizing the F1-score, Precision, and Recall using a Seaborn bar plot.")
        metrics = ["Precision", "Recall", "F1-Score"]
        classes = ["Class 0", "Class 1"]
        values = [[0.75, 0.87, 0.80], [0.72, 0.54, 0.62]]  # Values from screenshot
        metric_df = pd.DataFrame(values, columns=metrics, index=classes).reset_index()
        melted_metric_df = metric_df.melt(id_vars=["index"], var_name="Metric", value_name="Score")
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=melted_metric_df, x="Metric", y="Score", hue="index", palette="viridis")
        plt.title("Model Performance by Metric and Class")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.legend(title="Class")
        st.pyplot(plt)


# Page 3: EDA of Original Data Set
elif page == "EDA of Original Data Set":
    st.title("Exploratory Data Analysis (EDA) of Original Data Set")
    st.write("This page explores the dataset used to train the diabetes prediction model.")

    # Load the dataset
    diabetes_df = pd.read_csv("diabetes.csv")


    # Plot 1: Interactive scatter plot
    st.subheader("Scatter Plot 1: Explore Feature Relationships")

    # Dropdowns for selecting features for the first scatter plot
    feature_x = st.selectbox("Feature X (Scatter Plot 1)", diabetes_df.columns, key="scatter1_x")
    feature_y = st.selectbox("Feature Y (Scatter Plot 1)", diabetes_df.columns, key="scatter1_y")

    # Create the scatter plot
    fig1 = px.scatter(
        diabetes_df,
        x=feature_x,
        y=feature_y,
        color="Outcome",  # Predefined color parameter
        title=f"{feature_x} vs {feature_y}",
        labels={feature_x: feature_x, feature_y: feature_y}
    )
    st.plotly_chart(fig1)

    # Plot 2: Animated scatter plot
    # Sort the dataset by Age to ensure the animation is in order
    diabetes_df_sorted = diabetes_df.sort_values(by="Age")

    # Plot 2: Animated scatter plot with sorted Age
    st.subheader("Scatter Plot 2: Animated Feature Relationships")

    # Dropdowns for selecting features for the second animated scatter plot
    x_feature_animated = st.selectbox("X Feature (Scatter Plot 2)", diabetes_df.columns, key="scatter2_x")
    y_feature_animated = st.selectbox("Y Feature (Scatter Plot 2)", diabetes_df.columns, key="scatter2_y")

    # Create the animated scatter plot
    fig2 = px.scatter(
        diabetes_df_sorted,
        x=x_feature_animated,
        y=y_feature_animated,
        animation_frame="Age",  # Predefined animation parameter
        color="Outcome",  # Predefined color parameter
        size="BMI",  # Predefined size parameter
        title=f"{x_feature_animated} vs {y_feature_animated} Across Age",
        labels={x_feature_animated: x_feature_animated, y_feature_animated: y_feature_animated}
    )

    st.plotly_chart(fig2)


    # Show basic dataset info
    with st.expander("Dataset Information"):
        st.dataframe(diabetes_df.head(10))  # Display the first 10 rows of the dataset
        st.write("**Dataset Summary:**")
        st.write(diabetes_df.describe())
        st.write("**Dataset Info:**")
        buffer = StringIO()
        diabetes_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # Step 2: Visualize the Outcome Variable
    with st.expander("Outcome Variable Distribution"):
        outcome_count = diabetes_df["Outcome"].value_counts().reset_index()
        outcome_count.columns = ["Outcome", "Count"]
        fig_outcome = px.bar(outcome_count, x="Outcome", y="Count", title="Count of Outcome Variable",
                            labels={"Outcome": "Outcome", "Count": "Count"},
                            color_discrete_sequence=["blue"])
        st.plotly_chart(fig_outcome)

        # Percentage of Diabetes Diagnosed
        diabetes_true = diabetes_df[diabetes_df['Outcome'] == 1]
        diabetes_percentage = (len(diabetes_true) / len(diabetes_df)) * 100
        st.subheader("Diabetes Diagnosed Percentage")
        st.write(f"Diabetes diagnosed in percentage: **{diabetes_percentage:.2f}%**")

    # Heatmap of Correlation Matrix using Plotly
    with st.expander("Correlation Heatmap"):

        # Compute the correlation matrix
        correlation_matrix = diabetes_df.corr()

        # Create the heatmap using Plotly
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale="magma",
                text=correlation_matrix.values,  # Show values on hover
                texttemplate="%{text:.2f}",  # Format values to two decimal places
                showscale=True
            )
        )

        # Update layout for better aesthetics
        fig_corr.update_layout(
            title="Correlation Matrix of Features",
            xaxis=dict(title="Features"),
            yaxis=dict(title="Features"),
            font=dict(size=12),
            width=800,
            height=800
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig_corr)


    # Highest Correlating Features with Outcome
    with st.expander("Top Features Correlating with Outcome"):

        # Compute the correlation matrix
        correlation_matrix = diabetes_df.corr()

        # Sort the correlations of all features with the target variable 'Outcome'
        correlation_with_outcome = correlation_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)

        # Take the top n highest correlations
        top_features = correlation_with_outcome.nlargest(5)

        # Create a bar chart using Plotly
        fig_top_corr = px.bar(
            x=top_features.index,
            y=top_features.values,
            labels={'x': 'Features', 'y': 'Correlation Coefficient'},
            title="Top 5 Features Correlating with Outcome",
            text=top_features.values
        )

        # Update aesthetics
        fig_top_corr.update_traces(marker_color='indigo', texttemplate='%{text:.2f}', textposition='outside')
        fig_top_corr.update_layout(
            xaxis=dict(title="Features"),
            yaxis=dict(title="Correlation Coefficient"),
            title=dict(font=dict(size=20)),
            font=dict(size=14)
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig_top_corr)




    
# Apply styling for footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
        color: #555555;
    }
    </style>
    <div class="footer">
        Developed with ❤️ by Alejandro Fitzner López
    </div>
    """,
    unsafe_allow_html=True,
)
