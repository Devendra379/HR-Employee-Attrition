# %%writefile streamlit_app.py

# 1. Necessary Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pyngrok import ngrok
import subprocess
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel


# Set Streamlit page configuration - MUST be the first Streamlit command
st.set_page_config(layout="wide")

# Define file paths for saved artifacts
MODEL_RECALL_FILENAME = 'best_attrition_model.joblib'
THRESHOLD_RECALL_FILENAME = 'recall_tuned_threshold.joblib'

MODEL_F1_FILENAME = 'best_attrition_model_f1_tuned.joblib'
THRESHOLD_F1_FILENAME = 'best_attrition_threshold_f1_tuned.joblib'

MODEL_PRECISION_FILENAME = 'best_attrition_model_precision_tuned.joblib'
THRESHOLD_PRECISION_FILENAME = 'best_attrition_threshold_precision_tuned.joblib'

PREPROCESSOR_FILENAME = 'feature_selection_pipeline.joblib'
COLUMNS_BEFORE_PIPELINE_FILENAME = 'columns_before_pipeline_transform.joblib'
TRAINING_AGE_BUCKET_MODE_FILENAME = 'training_age_bucket_mode.joblib'


# 2. Load Saved Artifacts
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        # Load Recall-tuned artifacts
        try:
            if os.path.exists(MODEL_RECALL_FILENAME) and os.path.exists(THRESHOLD_RECALL_FILENAME):
                 artifacts['recall_model'] = joblib.load(MODEL_RECALL_FILENAME)
                 artifacts['recall_threshold'] = joblib.load(THRESHOLD_RECALL_FILENAME)
            else:
                 st.sidebar.warning(f"Recall-tuned model artifacts not found: {MODEL_RECALL_FILENAME}, {THRESHOLD_RECALL_FILENAME}")
        except Exception as e:
            st.sidebar.error(f"Error loading Recall-tuned model artifacts: {e}")

        # Load F1-tuned artifacts
        try:
            if os.path.exists(MODEL_F1_FILENAME) and os.path.exists(THRESHOLD_F1_FILENAME):
                 artifacts['f1_model'] = joblib.load(MODEL_F1_FILENAME)
                 artifacts['f1_threshold'] = joblib.load(THRESHOLD_F1_FILENAME)
            else:
                 st.sidebar.warning(f"F1-tuned model artifacts not found: {MODEL_F1_FILENAME}, {THRESHOLD_F1_FILENAME}")
        except Exception as e:
            st.sidebar.error(f"Error loading F1-tuned model artifacts: {e}")

        # Load Precision-tuned artifacts
        try:
            if os.path.exists(MODEL_PRECISION_FILENAME) and os.path.exists(THRESHOLD_PRECISION_FILENAME):
                artifacts['precision_model'] = joblib.load(MODEL_PRECISION_FILENAME)
                artifacts['precision_threshold'] = joblib.load(THRESHOLD_PRECISION_FILENAME)
            else:
                st.sidebar.warning(f"Precision-tuned model artifacts not found: {MODEL_PRECISION_FILENAME}, {THRESHOLD_PRECISION_FILENAME}")
        except Exception as e:
            st.sidebar.error(f"Error loading Precision-tuned model artifacts: {e}")

        # Load common preprocessing artifacts - these are critical
        artifacts['preprocessor_pipeline'] = joblib.load(PREPROCESSOR_FILENAME)
        artifacts['columns_before_pipeline'] = joblib.load(COLUMNS_BEFORE_PIPELINE_FILENAME)
        artifacts['training_age_bucket_mode'] = joblib.load(TRAINING_AGE_BUCKET_MODE_FILENAME)

        # Check if at least the preprocessor loaded
        if 'preprocessor_pipeline' in artifacts and artifacts['preprocessor_pipeline'] is not None:
             return artifacts
        else:
             st.error("Critical preprocessing pipeline failed to load.")
             return None

    except FileNotFoundError as e:
        st.error(f"Error loading a critical artifact: {e}. Make sure the preprocessing pipeline and columns list ({PREPROCESSOR_FILENAME}, {COLUMNS_BEFORE_PIPELINE_FILENAME}, {TRAINING_AGE_BUCKET_MODE_FILENAME}) are in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during artifact loading: {e}")
        return None

# Load artifacts when the app starts
loaded_artifacts = load_artifacts()


# 3. Define Preprocessing Function for New Data
def preprocess_input_data(input_df, preprocessor_pipeline, columns_before_pipeline, training_age_bucket_mode):
    """
    Applies the same feature engineering, initial column drops, and preprocessing
    steps as used during training to new input data.

    Args:
        input_df (pd.DataFrame): Raw input data as a pandas DataFrame (single row).
        preprocessor_pipeline: The fitted scikit-learn pipeline including preprocessing and selection.
        columns_before_pipeline (list): List of column names expected by the pipeline's fit method.
        training_age_bucket_mode (float or int): Mode of 'age_bucket_encoded' from training data for imputation.


    Returns:
        np.ndarray: Preprocessed and feature-selected data ready for model prediction.
    """
    df_processed = input_df.copy()

    # --- Reapply Feature Engineering steps (must match training exactly) ---
    # Use .loc[0] to access scalar values from the single-row DataFrame for calculations
    # Handle potential missing columns in input_df gracefully by providing default values
    # using .get() with pd.Series of default values and the same index as df_processed

    # Calculate new features
    # Ensure scalar values are used in calculations
    total_working_years = df_processed.get('TotalWorkingYears', pd.Series([0], index=df_processed.index)).iloc[0]
    years_at_company = df_processed.get('YearsAtCompany', pd.Series([0], index=df_processed.index)).iloc[0]
    job_satisfaction = df_processed.get('JobSatisfaction', pd.Series([0], index=df_processed.index)).iloc[0]
    environment_satisfaction = df_processed.get('EnvironmentSatisfaction', pd.Series([0], index=df_processed.index)).iloc[0]
    relationship_satisfaction = df_processed.get('RelationshipSatisfaction', pd.Series([0], index=df_processed.index)).iloc[0]
    over_time_encoded_val = df_processed.get('OverTime', pd.Series(['No'], index=df_processed.index)).map({'Yes': 1, 'No': 0}).fillna(0).iloc[0]
    job_level = df_processed.get('JobLevel', pd.Series([1], index=df_processed.index)).iloc[0]
    years_in_current_role = df_processed.get('YearsInCurrentRole', pd.Series([0], index=df_processed.index)).iloc[0]
    num_companies_worked = df_processed.get('NumCompaniesWorked', pd.Series([0], index=df_processed.index)).iloc[0]
    years_since_last_promotion = df_processed.get('YearsSinceLastPromotion', pd.Series([0], index=df_processed.index)).iloc[0]
    job_involvement = df_processed.get('JobInvolvement', pd.Series([0], index=df_processed.index)).iloc[0]
    monthly_income = df_processed.get('MonthlyIncome', pd.Series([0], index=df_processed.index)).iloc[0]
    age_val = df_processed.get('Age', pd.Series([0], index=df_processed.index)).iloc[0]
    business_travel_val = df_processed.get('BusinessTravel', pd.Series(['Travel_Rarely'], index=df_processed.index)).map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}).fillna(1).iloc[0]


    df_processed['years_at_other_companies'] = max(total_working_years - years_at_company, 0)
    df_processed['satisfaction_level'] = job_satisfaction + environment_satisfaction + relationship_satisfaction
    df_processed['over_time_encoded'] = over_time_encoded_val
    df_processed['career_progress'] = job_level / (total_working_years + 1)
    df_processed['stability_index'] = years_in_current_role / (years_at_company + 1)
    df_processed['avg_tenure_per_company'] = total_working_years / (num_companies_worked + 1)
    df_processed['is_hopper'] = (df_processed['avg_tenure_per_company'].iloc[0] < 2).astype(int) # Access scalar for comparison
    df_processed['is_recent_joiner'] = (years_at_company < 1).astype(int)
    df_processed['is_hopper_recent_joiner'] = (df_processed['is_hopper'].iloc[0] & df_processed['is_recent_joiner'].iloc[0]).astype(int) # Access scalars

    # Reapply Age Bucket feature engineering and mapping
    age_bins = [18, 30, 40, 50, 60, 61]
    age_labels = ['20s', '30s', '40s', '50s', '60+']
    if 'Age' in df_processed.columns:
        df_processed['age_bucket'] = pd.cut(df_processed['Age'], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
        age_map = {label: i+1 for i, label in enumerate(age_labels)}
        df_processed['age_bucket_encoded'] = df_processed['age_bucket'].map(age_map)
        if training_age_bucket_mode is not None:
             df_processed['age_bucket_encoded'].fillna(training_age_bucket_mode, inplace=True)
        else:
            df_processed['age_bucket_encoded'].fillna(0, inplace=True)
    else:
        df_processed['age_bucket_encoded'] = training_age_bucket_mode if training_age_bucket_mode is not None else 0

    # Reapply BusinessTravel mapping
    df_processed['business_travel_encoded'] = business_travel_val

    # Reapply Interaction terms (handle missing base columns)
    df_processed['overtime_job_satisfaction_interaction'] = over_time_encoded_val * job_satisfaction
    df_processed['job_involvement_job_satisfaction_interaction'] = job_involvement * job_satisfaction
    df_processed['environment_satisfaction_job_satisfaction_interaction'] = environment_satisfaction * job_satisfaction
    df_processed['overtime_years_since_last_promotion_interaction'] = over_time_encoded_val * years_since_last_promotion
    df_processed['overtime_satisfaction_level_interaction'] = over_time_encoded_val * df_processed['satisfaction_level'].iloc[0]
    df_processed['job_level_total_working_years_interaction'] = job_level * total_working_years
    df_processed['years_at_company_job_involvement_interaction'] = years_at_company * job_involvement
    df_processed['monthly_income_job_level_interaction'] = monthly_income * job_level

    # Reapply Polynomial features (handle missing base columns)
    df_processed['total_working_years_sq'] = total_working_years**2
    df_processed['monthly_income_sq'] = monthly_income**2
    df_processed['age_sq'] = age_val**2

    # --- Step 2: Drop original columns that were dropped during training *before* preprocessing pipeline ---
    cols_to_drop_original_before_pipeline = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Attrition', 'OverTime', 'BusinessTravel', 'age_bucket']
    cols_to_drop_exist = [col for col in cols_to_drop_original_before_pipeline if col in df_processed.columns]
    df_processed = df_processed.drop(cols_to_drop_exist, axis=1, errors='ignore')

    # --- Step 3: Ensure columns match the expected input for the pipeline ---
    missing_cols = set(columns_before_pipeline) - set(df_processed.columns)
    for col in missing_cols:
        df_processed[col] = 0.0

    df_final_for_pipeline = df_processed.reindex(columns=columns_before_pipeline, fill_value=0.0)

    # --- Step 4: Apply the fitted preprocessing pipeline ---
    processed_data = preprocessor_pipeline.transform(df_final_for_pipeline)
    return processed_data

# 5. Function to Get Feature Names After Preprocessing and Selection
def get_feature_names_after_pipeline(preprocessor_pipeline):
    """
    Attempts to get feature names after applying the preprocessing pipeline,
    especially considering one-hot encoding and feature selection.

    Args:
        preprocessor_pipeline: The fitted scikit-learn pipeline.

    Returns:
        list: A list of feature names after the pipeline, or None if unsuccessful.
    """
    feature_names = None
    try:
        # Check if the pipeline has a 'preprocessor' step (ColumnTransformer)
        if hasattr(preprocessor_pipeline, 'named_steps') and 'preprocessor' in preprocessor_pipeline.named_steps:
            preprocessor = preprocessor_pipeline.named_steps['preprocessor']
            # Get names after the ColumnTransformer
            if hasattr(preprocessor, 'get_feature_names_out'):
                 feature_names_after_preprocess = list(preprocessor.get_feature_names_out())
                 # Check if there's a 'selector' step (SelectFromModel)
                 if 'selector' in preprocessor_pipeline.named_steps and hasattr(preprocessor_pipeline.named_steps['selector'], 'get_support'):
                      selector = preprocessor_pipeline.named_steps['selector']
                      selector_mask = selector.get_support()
                      # Filter names based on the selector mask
                      # Ensure mask length matches feature names before filtering
                      if len(feature_names_after_preprocess) == len(selector_mask):
                           feature_names = [feature_names_after_preprocess[i] for i, keep in enumerate(selector_mask) if keep]
                      else:
                           # Mismatch, cannot reliably filter
                           feature_names = None # Indicate failure
                 else:
                      # If no selector, names are after the preprocessor
                      feature_names = feature_names_after_preprocess
            # Add checks for other potential structures or simpler pipelines
            elif hasattr(preprocessor_pipeline, 'get_feature_names_out'):
                 # If the whole pipeline has get_feature_names_out (less common for Pipelines with SelectFromModel)
                 feature_names = list(preprocessor_pipeline.get_feature_names_out())

    except Exception as e:
        st.sidebar.warning(f"Error attempting to get feature names from pipeline: {e}")
        feature_names = None # Indicate failure

    return feature_names


# 6. Function to Interpret Prediction based on Model Coefficients
def interpret_prediction(model, processed_input, preprocessor_pipeline, num_top_features=5):
    """
    Interprets the prediction of a Logistic Regression model based on its coefficients.
    Highlights features that contribute most to the positive prediction.

    Args:
        model: The fitted Logistic Regression model.
        processed_input (np.ndarray): The preprocessed and feature-selected input data (single row).
        preprocessor_pipeline: The fitted scikit-learn pipeline including preprocessing and selection.
        num_top_features (int): The number of top contributing features to display.

    Returns:
        list: A list of strings describing the top contributing factors.
    """
    interpretation = []

    if isinstance(model, LogisticRegression) and hasattr(model, 'coef_'):
        try:
            coefficients = model.coef_[0] # For binary classification

            # Get accurate feature names after the pipeline
            feature_names_after_pipeline = get_feature_names_after_pipeline(preprocessor_pipeline)


            # Fallback if getting accurate names failed or names don't match coefficients
            if feature_names_after_pipeline is None or len(feature_names_after_pipeline) != len(coefficients):
                 # Use generic names as a fallback
                 feature_names_after_pipeline = [f"Feature_{i}" for i in range(len(coefficients))]
                 interpretation.insert(0, "_Note: Could not retrieve accurate feature names from the pipeline. Displaying generic feature indices._")


            # Create a Series of coefficients with feature names
            feature_importance = pd.Series(coefficients, index=feature_names_after_pipeline)

            # Sort features by the absolute value of coefficients to find most impactful
            sorted_importance = feature_importance.abs().sort_values(ascending=False)

            interpretation.append("Top contributing factors to this prediction:")

            # Display top N most impactful features (both positive and negative influence)
            # Filter out features with exactly 0 coefficient if any resulted from L1 regularization or similar
            sorted_importance = sorted_importance[sorted_importance > 1e-9] # Filter near-zero coefficients

            top_impactful_features = sorted_importance.head(num_top_features).index.tolist()


            if not top_impactful_features:
                 interpretation.append("_No significant factors identified for interpretation._")
            else:
                 for feature in top_impactful_features:
                      if feature in feature_importance.index: # Ensure feature exists
                           coeff = feature_importance[feature]
                           # Determine direction of influence
                           influence_direction = "increase" if coeff > 0 else "decrease"

                           # Simplify feature names for better readability
                           simple_feature_name = feature.replace('num__', '').replace('cat__', '').replace('_encoded', '').replace('_interaction', ' Interaction')
                           simple_feature_name = simple_feature_name.replace('_', ' ') # Replace underscores with spaces

                           # Rephrase the message
                           interpretation.append(f"- **{simple_feature_name}**: Tends to **{influence_direction}** the likelihood of attrition.") # Removed coefficient for simplicity


        except Exception as e:
             interpretation.append(f"_Could not generate detailed interpretation: {e}_")
             interpretation.append("_Interpretation is currently best supported for Logistic Regression models._")

    else:
        interpretation.append("_Interpretation is currently best supported for Logistic Regression models._")

    return interpretation


# 4. Set up the Streamlit App Structure
def main():
    st.title("HR Employee Attrition Prediction")
    st.write("Enter employee details and select the prediction objective to predict the likelihood of attrition.")

    # Check if critical artifacts loaded successfully before proceeding
    if loaded_artifacts is None or loaded_artifacts.get('preprocessor_pipeline') is None:
        st.error("Critical model artifacts (preprocessing pipeline) not loaded. Please check the file paths and ensure the artifacts exist.")
        return

    # Extract necessary artifacts
    preprocessor_pipeline = loaded_artifacts.get('preprocessor_pipeline')
    columns_before_pipeline = loaded_artifacts.get('columns_before_pipeline')
    training_age_bucket_mode = loaded_artifacts.get('training_age_bucket_mode')


    # --- Model Selection ---
    st.sidebar.header("Prediction Objective")
    objective = st.sidebar.selectbox(
        "Select Prediction Objective:",
        options=[
            "🔍 High Recall – Catch All Risks",
            "✅ High Precision – Only Strong Signals",
            "⚖️ Balanced – Best F1 Score",
            "🤖 Smart Pick (Auto-Optimized)"
        ],
        help="Choose the prediction objective based on HR's priority."
    )

    # Select the model and threshold based on the objective
    selected_model = None
    selected_threshold = None
    objective_description = ""

    if objective == "🔍 High Recall – Catch All Risks":
        selected_model = loaded_artifacts.get('recall_model')
        selected_threshold = loaded_artifacts.get('recall_threshold')
        objective_description = "tuned to **maximize Recall** (minimize False Negatives)."
        if selected_model is None:
            st.sidebar.warning("Recall-tuned model not loaded. Please select another objective.")
            return

    elif objective == "✅ High Precision – Only Strong Signals":
        selected_model = loaded_artifacts.get('precision_model')
        selected_threshold = loaded_artifacts.get('precision_threshold')
        objective_description = "tuned to **maximize Precision** (minimize False Positives)."
        if selected_model is None:
            st.sidebar.warning("Precision-tuned model not loaded. Please select another objective.")
            return


    elif objective == "⚖️ Balanced – Best F1 Score":
        selected_model = loaded_artifacts.get('f1_model')
        selected_threshold = loaded_artifacts.get('f1_threshold')
        objective_description = "tuned to **maximize F1-score** (balance Precision and Recall)."
        if selected_model is None:
             st.sidebar.warning("F1-tuned model not loaded. Please select another objective.")
             return

    elif objective == "🤖 Smart Pick (Auto-Optimized)":
        selected_model = loaded_artifacts.get('f1_model')
        selected_threshold = loaded_artifacts.get('f1_threshold')
        objective_description = "using a **Smart Pick** strategy (currently defaults to F1-tuned)."
        if selected_model is None:
             st.sidebar.warning("Smart Pick model (defaults to F1-tuned) not loaded. Please select another objective.")
             return


    # Check if a model and threshold were successfully selected/loaded for the chosen objective
    if selected_model is None or selected_threshold is None:
        st.error(f"Could not load the necessary model or threshold for the '{objective}' objective. Please ensure the corresponding artifact files exist or select another objective.")
        return


    st.header("Employee Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal & Role Details")
        age = st.number_input("Age", min_value=18, max_value=60, value=30, help="Employee's age")
        gender = st.selectbox("Gender", options=['Female', 'Male'], help="Employee's gender")
        marital_status = st.selectbox("Marital Status", options=['Single', 'Married', 'Divorced'], help="Employee's marital status")
        department = st.selectbox("Department", options=['Sales', 'Research & Development', 'Human Resources'], help="Department the employee works in")

        # Define job roles based on department
        job_roles_by_department = {
            'Sales': ['Sales Executive', 'Sales Representative'],
            'Research & Development': ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Research Director'],
            'Human Resources': ['Human Resources']
        }

        # Filter job roles based on selected department
        available_job_roles = job_roles_by_department.get(department, []) # Get roles for selected department, default to empty list if department not found

        # Add 'Manager' job role as it can exist in multiple departments (though often associated with Job Level 4/5)
        # For simplicity here, let's add Manager as an option if it's not already in the filtered list,
        # or handle it separately if it's a distinct role across departments.
        # Based on the original data, Manager is a separate JobRole.
        # Let's keep the original list of job roles and filter based on department mapping.
        # A better approach might be to use a multi-select for department and then list all roles.
        # For now, let's stick to filtering the single Job Role selectbox based on the single Department selectbox.

        # Let's refine the job role filtering based on the common roles per department from the data
        all_job_roles = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']
        filtered_job_roles = []
        if department == 'Sales':
            filtered_job_roles = ['Sales Executive', 'Sales Representative']
        elif department == 'Research & Development':
             filtered_job_roles = ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Research Director']
        elif department == 'Human Resources':
             filtered_job_roles = ['Human Resources']

        # Add 'Manager' as an option if it was present in the original data and is relevant (e.g., level > 3)
        # This is a simplification; a more robust mapping would be needed for production.
        # For this example, let's just include 'Manager' as an option if the department is R&D or Sales,
        # where managers are typically found in the original dataset.
        if department in ['Sales', 'Research & Development']:
             if 'Manager' not in filtered_job_roles:
                 filtered_job_roles.append('Manager')


        job_role = st.selectbox("Job Role", options=filtered_job_roles, help="Job role of the employee (filtered by department)")

        job_level = st.selectbox("Job Level", options=[1, 2, 3, 4, 5], format_func=lambda x: f"Level {x}", help="Job level of the employee (1 to 5)")
        # Updated Education options
        education_options = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        education_display = st.selectbox("Education", options=list(education_options.keys()), format_func=lambda x: education_options[x], help="Education level")
        education = education_display # Use the selected numerical value


        education_field = st.selectbox("Education Field", options=['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'], help="Field of education")


    with col2:
        st.subheader("Work & Satisfaction Metrics")
        distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=29, value=5, help="Distance from home to work (miles)")
        business_travel = st.selectbox("Business Travel", options=['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], help="Frequency of business travel")

        # Updated EnvironmentSatisfaction options
        environment_satisfaction_options = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        environment_satisfaction_display = st.selectbox("Environment Satisfaction", options=list(environment_satisfaction_options.keys()), format_func=lambda x: environment_satisfaction_options[x], help="Environment satisfaction")
        environment_satisfaction = environment_satisfaction_display # Use the selected numerical value

        # Updated JobInvolvement options
        job_involvement_options = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        job_involvement_display = st.selectbox("Job Involvement", options=list(job_involvement_options.keys()), format_func=lambda x: job_involvement_options[x], help="Job involvement")
        job_involvement = job_involvement_display # Use the selected numerical value

        # Updated JobSatisfaction options
        job_satisfaction_options = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        job_satisfaction_display = st.selectbox("Job Satisfaction", options=list(job_satisfaction_options.keys()), format_func=lambda x: job_satisfaction_options[x], help="Job satisfaction")
        job_satisfaction = job_satisfaction_display # Use the selected numerical value

        # Updated RelationshipSatisfaction options
        relationship_satisfaction_options = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        relationship_satisfaction_display = st.selectbox("Relationship Satisfaction", options=list(relationship_satisfaction_options.keys()), format_func=lambda x: relationship_satisfaction_options[x], help="Relationship satisfaction")
        relationship_satisfaction = relationship_satisfaction_display # Use the selected numerical value

        # Updated WorkLifeBalance options
        work_life_balance_options = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
        work_life_balance_display = st.selectbox("Work Life Balance", options=list(work_life_balance_options.keys()), format_func=lambda x: work_life_balance_options[x], help="Work-life balance")
        work_life_balance = work_life_balance_display # Use the selected numerical value

        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800, help="Daily payment rate")
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65, help="Hourly payment rate")


    with col3:
        st.subheader("Experience & Compensation")
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=6000, help="Monthly income")
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000, help="Monthly rate")
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=11, max_value=25, value=15, help="Percentage salary hike for the year")

        # Updated PerformanceRating options
        performance_rating_options = {3: 'Excellent', 4: 'Outstanding'}
        performance_rating_display = st.selectbox("Performance Rating", options=list(performance_rating_options.keys()), format_func=lambda x: performance_rating_options[x], help="Performance rating")
        performance_rating = performance_rating_display # Use the selected numerical value

        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10, help="Total years of working experience")
        years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5, help="Total years spent at the current company")
        years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=18, value=3, help="Years in current role at the company")
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1, help="Years since last promotion")
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=17, value=3, help="Years with current manager")
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=9, value=1, help="Number of companies worked before the current one")
        stock_option_level = st.selectbox("Stock Option Level", options=[0, 1, 2, 3], help="Stock option level (0 to 3)")
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2, help="Number of times training was conducted last year")
        over_time = st.selectbox("Over Time", options=['Yes', 'No'], help="Whether the employee works overtime")

    employee_count = 1
    employee_number = 9999
    over18 = 'Y'
    standard_hours = 80

    if st.button("Predict Attrition"):
        input_data_dict = {
            'Age': [age],
            'DailyRate': [daily_rate],
            'DistanceFromHome': [distance_from_home],
            'Education': [education], # Use the selected numerical value
            'EnvironmentSatisfaction': [environment_satisfaction], # Use the selected numerical value
            'HourlyRate': [hourly_rate],
            'JobInvolvement': [job_involvement], # Use the selected numerical value
            'JobLevel': [job_level],
            'JobSatisfaction': [job_satisfaction], # Use the selected numerical value
            'MaritalStatus': [marital_status],
            'MonthlyIncome': [monthly_income],
            'MonthlyRate': [monthly_rate],
            'NumCompaniesWorked': [num_companies_worked],
            'OverTime': [over_time],
            'PercentSalaryHike': [percent_salary_hike],
            'PerformanceRating': [performance_rating], # Use the selected numerical value
            'RelationshipSatisfaction': [relationship_satisfaction], # Use the selected numerical value
            'StockOptionLevel': [stock_option_level],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_times_last_year],
            'WorkLifeBalance': [work_life_balance], # Use the selected numerical value
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'YearsWithCurrManager': [years_with_curr_manager],
            'BusinessTravel': [business_travel],
            'Department': [department],
            'EducationField': [education_field],
            'Gender': [gender],
            'JobRole': [job_role],
            'EmployeeCount': [employee_count],
            'EmployeeNumber': [employee_number],
            'Over18': [over18],
            'StandardHours': [standard_hours]
        }

        input_df = pd.DataFrame(input_data_dict)

        try:
            processed_input = preprocess_input_data(
                input_df,
                preprocessor_pipeline,
                loaded_artifacts.get('columns_before_pipeline'),
                loaded_artifacts.get('training_age_bucket_mode')
            )

            prediction_proba = selected_model.predict_proba(processed_input)[:, 1]
            final_prediction = (prediction_proba >= selected_threshold).astype(int)

            st.subheader("Prediction Result")
            st.write(f"Probability of Attrition: {prediction_proba[0]:.4f}")

            if final_prediction[0] == 1:
                st.error("Prediction: Employee is likely to attrite.")
            else:
                st.success("Prediction: Employee is unlikely to attrite.")

            st.info(f"*(This prediction is based on the '{objective}' objective, using a threshold of {selected_threshold:.4f}.)*")

            if objective == "🔍 High Recall – Catch All Risks":
                 st.warning("*(Note: The 'High Recall' objective is designed to catch as many potential leavers as possible, which may result in more employees being flagged for review who ultimately do not attrite.)*")


            # --- Display Potential Reasons (Interpretation) ---
            if final_prediction[0] == 1:
                 st.subheader("Potential Reasons for Predicted Attrition")
                 interpretation_messages = interpret_prediction(
                     selected_model,
                     processed_input,
                     preprocessor_pipeline
                 )
                 for message in interpretation_messages:
                      st.markdown(message)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check the input values and ensure all required artifacts are loaded correctly.")

# Run the app
if __name__ == "__main__":
    main()