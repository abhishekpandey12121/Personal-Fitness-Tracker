import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import base64

import warnings
warnings.filterwarnings('ignore')

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Encode your background image and logo
background_image_base64 = encode_image_to_base64("C:/Users/abhishek pandey/Desktop/Implementation of Personal Fitness Tracker using Python/fitness.jpg")
logo_base64 = encode_image_to_base64("C:/Users/abhishek pandey/Desktop/Implementation of Personal Fitness Tracker using Python/logo.jpg")

# Custom CSS for styling
custom_css = """
<style>
body {
    background-image: url('data:image/jpeg,{background_image_base64}'); /* Replace with your image URL */
    background-size: cover;
    background-attachment: fixed;
}

.sidebar .sidebar-content {
    background-image: url('data:data:image/jpeg,{background_image_base64}'); /* Replace with your image URL */
    background-size: cover;
    background-attachment: fixed;
}

.logo {
    position: fixed;
    top: 10px;
    left: 10px;
    width: 100px;
}
</style>
"""

# Inject CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Logo image (replace 'logo.png' with your logo file path)
st.markdown(f'<img class="logo" src="data:image/png;base64,{logo_base64}">', unsafe_allow_html=True)

# Chatbot function
def chatbot(user_input):
    user_input = user_input.lower()
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you today?"
    elif "calories" in user_input or "predict" in user_input:
        return "You can predict your calories burned by entering your details in the sidebar."
    elif "features" in user_input or "parameters" in user_input:
        return "The features used for prediction are Age, Gender, BMI, Duration, Heart Rate, and Body Temperature."
    elif "feedback" in user_input:
        return "You can provide feedback at the bottom of the page."
    elif "language" in user_input:
        return "This app supports English and Spanish. Use the selector at the top to switch languages."
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Bilingual support
def translate(text, language):
    translations = {
        "es": {
            "Personal Fitness Tracker": "Rastreador de Fitness Personal",
            "In this WebApp you will be able to observe your predicted calories burned in your body.": "En esta WebApp podrás observar las calorías predichas quemadas en tu cuerpo.",
            "User Input Parameters: ": "Parámetros de Entrada del Usuario: ",
            "Age: ": "Edad: ",
            "BMI: ": "IMC: ",
            "Duration (min): ": "Duración (min): ",
            "Heart Rate: ": "Frecuencia Cardíaca: ",
            "Body Temperature (C): ": "Temperatura Corporal (C): ",
            "Gender: ": "Género: ",
            "Your Parameters: ": "Sus Parámetros: ",
            "Prediction: ": "Predicción: ",
            "kilocalories": "kilocalorías",
            "Similar Results: ": "Resultados Similares: ",
            "General Information: ": "Información General: ",
            "You are older than": "Eres mayor que",
            "Your exercise duration is higher than": "Tu duración de ejercicio es mayor que",
            "You have a higher heart rate than": "Tienes una frecuencia cardíaca más alta que",
            "You have a higher body temperature than": "Tienes una temperatura corporal más alta que",
            "Feedback": "Comentarios",
            "Please provide your feedback:": "Por favor, proporcione su opinión:",
            "Submit Feedback": "Enviar Comentarios",
            "Thank you for your feedback!": "¡Gracias por su opinión!",
            "Please provide feedback before submitting.": "Por favor, proporcione comentarios antes de enviar."
        }
    }
    if language in translations:
        return translations[language].get(text, text)
    else:
        return text

# Streamlit app
st.write("## Personal Fitness Tracker")
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

# Language selection
language = st.selectbox("Select Language", ["English", "Español"])

# Translate the UI based on selected language
if language == "Español":
    st.write(translate("## Personal Fitness Tracker", "es"))
    st.write(translate("In this WebApp you will be able to observe your predicted calories burned in your body.", "es"))

st.sidebar.header(translate("User Input Parameters: ", language))

def user_input_features():
    age = st.sidebar.slider(translate("Age: ", language), 10, 100, 30)
    bmi = st.sidebar.slider(translate("BMI: ", language), 15, 40, 20)
    duration = st.sidebar.slider(translate("Duration (min): ", language), 0, 35, 15)
    heart_rate = st.sidebar.slider(translate("Heart Rate: ", language), 60, 130, 80)
    body_temp = st.sidebar.slider(translate("Body Temperature (C): ", language), 36, 42, 38)
    gender_button = st.sidebar.radio(translate("Gender: ", language), (translate("Male", language), translate("Female", language)))

    gender = 1 if gender_button == translate("Male", language) else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header(translate("Your Parameters: ", language))
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("C:/Users/abhishek pandey/Desktop/Implementation of Personal Fitness Tracker using Python/calories.csv")
exercise = pd.read_csv("C:/Users/abhishek pandey/Desktop/Implementation of Personal Fitness Tracker using Python/exercise.csv")

# Feature Engineering
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df['BMI'] = exercise_df['Weight'] / ((exercise_df['Height'] / 100) ** 2)
exercise_df['BMI'] = round(exercise_df['BMI'], 2)

# Age Categorization
def categorize_age(age):
    if 20 <= age < 40:
        return 'Young'
    elif 40 <= age < 60:
        return 'Middle-Aged'
    else:
        return 'Old'

exercise_df['Age_Category'] = exercise_df['Age'].apply(categorize_age)
exercise_df = pd.get_dummies(exercise_df, columns=['Age_Category', 'Gender'], drop_first=True)

# Ensure 'Gender_male' column exists
if 'Gender_male' not in exercise_df.columns:
    st.error(translate("Gender_male column not found in the dataset.", language))
else:
    # Visualizations
    def plot_feature_distributions(df):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sns.histplot(df['Age'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title(translate('Age Distribution', language))

        sns.histplot(df['BMI'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title(translate('BMI Distribution', language))

        sns.histplot(df['Duration'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title(translate('Duration Distribution', language))

        sns.histplot(df['Calories'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title(translate('Calories Distribution', language))

        plt.tight_layout()
        st.pyplot(fig)

    def plot_correlation_heatmap(df):
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title(translate('Correlation Heatmap', language))
        st.pyplot(fig)

    plot_feature_distributions(exercise_df)
    plot_correlation_heatmap(exercise_df)

    # Prepare the training and testing sets
    exercise_df.drop(columns="User_ID", inplace=True)

    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    # Prepare the training and testing sets
    exercise_train_data = exercise_train_data[["Gender_male", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender_male", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]

    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]

    # Train the model
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)

    # Align prediction data columns with training data
    df = df.reindex(columns=X_train.columns, fill_value=0)

    # Make prediction
    prediction = random_reg.predict(df)

    st.write("---")
    st.header(translate("Prediction: ", language))
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

    st.write(f"{round(prediction[0], 2)} **{translate('kilocalories', language)}**")

    st.write("---")
    st.header(translate("Similar Results: ", language))
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

    # Find similar results based on predicted calories
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    st.write(similar_data.sample(5))

    st.write("---")
    st.header(translate("General Information: ", language))

    # Boolean logic for age, duration, etc., compared to the user's input
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

    st.write(translate("You are older than", language), round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
    st.write(translate("Your exercise duration is higher than", language), round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
    st.write(translate("You have a higher heart rate than", language), round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
    st.write(translate("You have a higher body temperature than", language), round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")

    # Feedback section
    st.header(translate("Feedback", language))
    feedback = st.text_area(translate("Please provide your feedback:", language))

    if st.button(translate("Submit Feedback", language)):
        if feedback:
            st.success(translate("Thank you for your feedback!", language))
        else:
            st.warning(translate("Please provide feedback before submitting.", language))

    # Chatbot section
    st.header(translate("Chatbot", language))
    user_query = st.text_input(translate("Ask me anything about the app:", language))
    if user_query:
        response = chatbot(user_query)
        st.write(translate(response, language))
