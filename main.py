import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import streamlit as st

df = pd.read_csv("iris.csv")
model = joblib.load("model.pkl")

# Set variable for slider
st.title("Model Classification")
sl_min = df['sepal_length'].min()
sl_max = df['sepal_length'].max()
sl_med = (sl_max + sl_min) /2

sw_min = df['sepal_width'].min()
sw_max = df['sepal_width'].max()

pl_min = df['petal_length'].min()
pl_max = df['petal_length'].max()

pw_min = df['petal_width'].min()
pw_max = df['petal_width'].max()


selected_sl = st.slider('Select Sepal Length', min_value=float(sl_min), max_value=float(sl_max), value=float(sl_med))
selected_sw = st.slider('Select Sepal Width', min_value=float(sw_min), max_value=float(sw_max), value=float(sw_min))
selected_pl = st.slider('Select Petal Length', min_value=float(pl_min), max_value=float(pl_max), value=float(pl_min))
selected_pw = st.slider('Select Petal Width', min_value=float(pw_min), max_value=float(pw_max), value=float(pw_min))

data = pd.DataFrame({
    'sepal_length' : [selected_sl],
    'sepal_width' : [selected_sw],
    'petal_length': [selected_pl], 
    'petal_width': [selected_pw], 
})

predict = model.predict(data)[0]
if predict == 0:
    st.write("Setosa")
elif predict == 1:
    st.write("Versicolor")
else:
    st.write("Virginica")

# Set variable for select
st.title('Scatterplot')
columns = df.drop(columns='species', axis=1).columns.tolist()
columns.insert(0, "---Select Axis---")
x_axis = st.selectbox('Select X-Axis', columns)
y_axis = st.selectbox('Select Y-Axis', columns)
if x_axis != "---Select Axis---" and y_axis != "---Select Axis---":
    st.subheader("Select Theme")
    df = df
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="species",
        color_continuous_scale="reds",
    )

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)
else:
    st.write("Select Valid Axis")


