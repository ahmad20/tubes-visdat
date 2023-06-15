import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

# Load the dataset
df = pd.read_csv("iris.csv")

# Load the pre-trained model
model = joblib.load("model.pkl")

# Set page title and layout
st.set_page_config(page_title="Model Classification", layout="wide")

# Set variable for sliders
st.title("Model Classification")

# Calculate slider default values
slider_defaults = {
    'sepal_length': (df['sepal_length'].min() + df['sepal_length'].max()) / 2,
    'sepal_width': df['sepal_width'].min(),
    'petal_length': df['petal_length'].min(),
    'petal_width': df['petal_width'].min()
}

# Render sliders for feature selection
selected_sl = st.slider('Select Sepal Length', 
                        min_value=float(df['sepal_length'].min()),
                        max_value=float(df['sepal_length'].max()), 
                        value=float(slider_defaults['sepal_length']), step=0.1)
selected_sw = st.slider('Select Sepal Width', 
                        min_value=float(df['sepal_width'].min()),
                        max_value=float(df['sepal_width'].max()), 
                        value=float(slider_defaults['sepal_width']), step=0.1)
selected_pl = st.slider('Select Petal Length', 
                        min_value=float(df['petal_length'].min()),
                        max_value=float(df['petal_length'].max()), 
                        value=float(slider_defaults['petal_length']), step=0.1)
selected_pw = st.slider('Select Petal Width', 
                        min_value=float(df['petal_width'].min()),
                        max_value=float(df['petal_width'].max()), 
                        value=float(slider_defaults['petal_width']), step=0.1)

# Create a DataFrame with selected feature values
data = pd.DataFrame({
    'sepal_length': [selected_sl],
    'sepal_width': [selected_sw],
    'petal_length': [selected_pl],
    'petal_width': [selected_pw]
})

# Make predictions
prediction = model.predict(data)[0]

# Display predicted class
st.subheader('Prediction:')
if prediction == 0:
    st.write("Setosa")
elif prediction == 1:
    st.write("Versicolor")
else:
    st.write("Virginica")

# Set variable for scatterplot
st.title('Scatterplot')

# Select X and Y axes
columns = df.drop(columns='species').columns.tolist()
x_axis = st.selectbox('Select X-Axis', columns)
y_axis = st.selectbox('Select Y-Axis', columns)

if x_axis != y_axis:
    # Create scatterplot
    fig = px.scatter(df, x=x_axis, y=y_axis, color="species", color_continuous_scale="reds")

    # Display scatterplot with selected theme
    with st.expander("Select Theme"):
        theme = st.selectbox("Select Theme", ["streamlit", None])
        st.plotly_chart(fig, use_container_width=True, theme=theme)
else:
    st.write("Select different axes to generate the scatterplot.")
