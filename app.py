import streamlit as st
import requests
import google.generativeai as genai
import numpy as np
import xgboost as xgb

genai.configure(api_key="")

# Set up the app with multi-page navigation
st.set_page_config(page_title="Heart Health App", layout="wide")

model = xgb.XGBClassifier()
model.load_model(r"D:\Sem 7\Project\project_model.json")

def about_page():
    st.title("Heart Health Companion")
    st.markdown("""
    ## About the App
    This app uses advanced machine learning algorithms and a dataset of historical patient records to predict the likelihood of various heart diseases with improved accuracy.
    It enables healthcare providers and individuals to identify potential risks early, offering valuable insights to take proactive measures, adopt healthier lifestyles, and seek timely medical intervention.
    Predicting these conditions at an early stage can significantly reduce complications, improve treatment outcomes, and enhance quality of life.
    This app aims to bridge the gap between data-driven insights and practical healthcare solutions, empowering users to prioritize heart health.
    """)
    st.image("https://aanmc.org/wp-content/uploads/2017/12/iStock-628328284.jpg", caption="Heart Health")

def prediction_page():
    def predict_heart_disease(inputs):
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction

    st.title('Heart Disease Prediction')
    st.write("""
        Enter the following parameters to predict the likelihood of heart disease. 
        Hover over each field for detailed explanations. 
    """)

    age = st.number_input(
        'Age',
        min_value=20,
        max_value=120,
        step=1,
        format='%d'
    )
    if age < 20 or age > 120:
        st.error("Age must be between 20 and 120.")

    sex = st.selectbox(
        'Sex',
        options=["Choose an option", "Male", "Female"]
    )
    sex_mapping = {"Male": 1, "Female": 0}
    sex = sex_mapping.get(sex, None)

    cp = st.selectbox(
        'Chest Pain Type',
        options=["Choose an option", "Typical Angina", "Atypical Angina", "Non-anginal pain", "Asymptomatic"],
        help="""Typical Angina: Chest pain related to decreased blood flow to the heart.
                Atypical Angina: Chest pain not typical for decreased blood flow.
                Non-anginal pain: Chest pain not related to the heart.
                Asymptomatic: No chest pain."""
    )
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
    cp = cp_mapping.get(cp, None)

    trestbps = st.number_input(
        'Resting Blood Pressure',
        min_value=50,
        max_value=200,
        step=1,
        format='%d',
        help="Enter resting blood pressure in mm Hg. Must be between 50 and 200."
    )
    if trestbps < 50 or trestbps > 200:
        st.error("Resting blood pressure must be between 50 and 200.")

    chol = st.number_input(
        'Serum Cholesterol in mg/dl',
        min_value=100,
        max_value=600,
        step=1,
        format='%d',
        help="Enter serum cholesterol levels in mg/dl. Must be between 100 and 600."
    )
    if chol < 100 or chol > 600:
        st.error("Serum cholesterol must be between 100 and 600.")

    fbs = st.selectbox(
        'Fasting Blood Sugar > 120 mg/dl?',
        options=["Choose an option", "Yes", "No"]
    )
    fbs = 1 if fbs == "Yes" else (0 if fbs == "No" else None)

    restecg = st.selectbox(
        'Resting Electrocardiographic Results',
        options=["Choose an option", "Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
        help="""Normal: No ECG abnormalities.
                ST-T wave abnormality: May indicate ischemia or myocardial infarction.
                Left ventricular hypertrophy: Enlargement of the heart's left ventricle."""
    )
    restecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    restecg = restecg_mapping.get(restecg, None)

    thalch = st.number_input(
        'Maximum Heart Rate Achieved',
        min_value=50,
        max_value=220,
        step=1,
        format='%d',
        help="Enter the maximum heart rate achieved during exercise. Must be between 50 and 220."
    )
    if thalch < 50 or thalch > 220:
        st.error("Maximum heart rate must be between 50 and 220.")

    exang = st.selectbox(
        'Exercise Induced Angina',
        options=["Choose an option", "Yes", "No"],
        help="Yes: Chest pain induced by exercise. No: No chest pain during exercise."
    )
    exang = 1 if exang == "Yes" else (0 if exang == "No" else None)

    oldpeak = st.number_input(
        'Depression Induced by Exercise Relative to Rest',
        min_value=0.0,
        max_value=6.0,
        step=0.1,
        format='%f',
        help="ST depression induced by exercise relative to rest. Must be between 0.0 and 6.0."
    )
    if oldpeak < 0.0 or oldpeak > 6.0:
        st.error("Depression must be between 0.0 and 6.0.")

    slope = st.selectbox(
        'Slope of Peak Exercise ST Segment',
        options=["Choose an option", "Upsloping", "Flat", "Downsloping"],
        help="""Upsloping: Better heart function during exercise.
                Flat: No significant change in heart function.
                Downsloping: Potentially worsening heart function."""
    )
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_mapping.get(slope, None)

    ca = st.selectbox(
        'Number of Major Vessels Colored by Fluoroscopy',
        options=["Choose an option", "0", "1", "2", "3", "4"]
    )
    ca = int(ca) if ca.isdigit() else None

    thal = st.selectbox(
        'Thalassemia',
        options=["Choose an option", "Normal", "Fixed defect", "Reversible defect"],
        help="""Normal: No thalassemia.
                Fixed defect: Past myocardial infarction.
                Reversible defect: Ischemia detected during testing."""
    )
    thal_mapping = {"Normal": 0, "Fixed defect": 1, "Reversible defect": 2}
    thal = thal_mapping.get(thal, None)

    dataset = st.selectbox('Dataset', options=["Choose an option", 'Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'], help="Choose the relevant dataset")
    dataset_mapping = {"Cleveland": 0, "Hungary": 1, "Switzerland": 2,"VA Long Beach": 3}
    dataset = dataset_mapping.get(dataset, None)
    feature_14 = 0.0 

    inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, dataset, feature_14]

    # Submit button
    submit_button = st.button('Submit')

    if submit_button:
        if None not in inputs:
            prediction = predict_heart_disease(inputs)
            if (prediction == 1) or (prediction == 2) or (prediction == 3) or (prediction == 4):
                st.success(f"The person is at risk of heart disease of level {prediction}.")
            else:
                st.success("The person is not at risk of heart disease.")
        else:
            st.warning("Please fill out all fields before submitting.")

def exercises_page():
    st.subheader("Recommended Exercises for Heart Disease Patients")

    api_url = "https://exercisedb.p.rapidapi.com/exercises"

    headers = {
        'x-rapidapi-host': 'exercisedb.p.rapidapi.com',
        'x-rapidapi-key': '' 
    }
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        exercise_data = response.json()
        if isinstance(exercise_data, list) and len(exercise_data) > 0:
            for exercise in exercise_data:
                name = exercise.get('name', 'No name available')
                body_part = exercise.get('bodyPart', 'No body part listed')
                equipment = exercise.get('equipment', 'No equipment listed')
                target = exercise.get('target', 'No target muscle listed')
                instructions = exercise.get('instructions', [])
                gif_url = exercise.get('gifUrl', '')

                #Displaying exercise details
                st.write(f"**{name}**")
                st.write(f"**Target Muscle**: {target}")
                st.write(f"**Body Part**: {body_part}")
                st.write(f"**Equipment**: {equipment}")
                st.write(f"**Instructions**:")
                for i, instruction in enumerate(instructions, 1):
                    st.write(f"{i}. {instruction}")
                
                if gif_url:
                    st.image(gif_url, use_column_width=True)

                st.write("---")
        else:
            st.write("No exercises found in the response.")
    else:
        st.write(f"Failed to fetch data. Status Code: {response.status_code}")

def diet_page():
    app_id = '955037bf'
    app_key = ''
    api_url = "https://api.edamam.com/api/nutrition-details"
    st.title("Nutritional Analysis for Recipes")

    recipe_title = st.text_input("Enter Recipe Title", "Healthy Salad")

    ingredients_input = st.text_area("Enter Recipe Ingredients", "1 cup spinach, 1/2 cup tomatoes, 1 tablespoon olive oil, 1/2 avocado, 1 tablespoon lemon juice")

    ingredients = [ingredient.strip() for ingredient in ingredients_input.split(",")]
    if st.button('Analyze Nutritional Information'):
        if not ingredients:
            st.error("Please enter some ingredients.")
        else:
            data = {
                'ingr': ingredients  
            }

            url_with_credentials = f"{api_url}?app_id={app_id}&app_key={app_key}"
            response = requests.post(url_with_credentials, json=data)

            if response.status_code == 200:
                data = response.json()

                if 'totalNutrients' in data:
                    nutrients = data['totalNutrients']

                    st.write(f"**Nutritional Information for {recipe_title}**")

                    for nutrient in nutrients:
                        nutrient_name = nutrients[nutrient]['label']
                        nutrient_value = nutrients[nutrient]['quantity']
                        nutrient_unit = nutrients[nutrient]['unit']
                        st.write(f"{nutrient_name}: {nutrient_value:.2f} {nutrient_unit}")
                else:
                    st.write("No nutritional information available.")
            else:
                st.write(f"Failed to fetch data. Status Code: {response.status_code}")
                st.write(f"Error: {response.text}")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ("About", "Heart Disease Prediction", "Exercises", "Nutritional Analysis"))

    if page == "About":
        about_page()
    elif page == "Heart Disease Prediction":
        prediction_page()
    elif page == "Exercises":
        exercises_page()
    elif page == "Nutritional Analysis":
        diet_page() 

if __name__ == "__main__":
    main()
