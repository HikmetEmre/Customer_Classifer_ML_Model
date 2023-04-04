import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st

#### Page Config ###
st.set_page_config(
    page_title = "Customer Classifier",
    page_icon = "https://miro.medium.com/v2/resize:fit:2400/1*rGi8_JUoGX0L3W6nivmIAg@2x.png",
    menu_items={
        "Get help": "hikmetemreguler@gmail.com",
        "About": "For More Information\n" + "https://github.com/HikmetEmre/Project_3"
    }
)

### Title of Project ###
st.title("Customer Classification Project")

### Markdown ###
st.markdown("A big bussines wants to identify customers as  **:red[High Profile]** or **:yellow[Medium Profile]** or **:blue[Low Profile]**  by looking at the various features of the Customers have.")

### Adding Image ###
st.image("https://datafloq.com/wp-content/uploads/2021/12/blog_pictures2Fcustomer_segment-scaled.jpg")

st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a **machine learning model** in line with their needs and help them with their research.")
st.markdown("In addition, when they have information about a new Customer, they want us to come up with a product that we can predict Customer Profile Value based on this information.")
st.markdown("*Alright, Let's help them!*")

st.image("https://resources.pollfish.com/wp-content/uploads/2020/11/MARKET_RESEARCH_FOR_REAL_ESTATE_IN_CONTENT_1.png")

#### Header and definition of columns ###
st.header("META DATA")

st.markdown("- **CUSTOMER_ID**: Unique Num For Each Customer")
st.markdown("- **RECENCY**: A customer's last purchase was?")
st.markdown("- **FREQUENCY**:How often Customers bought")
st.markdown("- **REVENUE**: Total Amount of Customer's spending")
st.markdown("- **PROFILE**: What is the Customer's Profile? (High, Medium or Low")

### Example DF ON STREAMLIT PAGE ###
df=pd.read_csv("your_labeled_data_file.csv")


### Example TABLE ###
st.table(df.sample(5, random_state=18))

#---------------------------------------------------------------------------------------------------------------------

### Sidebar Markdown ###
st.sidebar.markdown("**INPUT** the features below to see the result!")

### Define Sidebar Input's ###
customer_id = st.sidebar.number_input("Unique Id Num", min_value=0)
Recency = st.sidebar.number_input("A customer's last purchase was.", min_value=0)
Frequency = st.sidebar.number_input("How often Customers bought", min_value=0)
Revenue = st.sidebar.number_input("Total Amount of Customer's spending.", min_value=0)


#---------------------------------------------------------------------------------------------------------------------

### Recall Model ###
from joblib import load

rndmfrst_model = load('random_forest_model.pkl')

input_df = pd.DataFrame({
    'Recency': [Recency],
    'Frequency': [Frequency],
    'Revenue': [Revenue]
})
### Scale the new input data###
scaler = StandardScaler()
input_df=scaler.fit_transform(input_df)

pred = rndmfrst_model.predict(input_df.values)


#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

### Result Screen ###
if st.sidebar.button("Submit"):

    ### Info message ###
    st.info("You can find the result below.")

    ### Inquiry Time Info ###
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    ### For showing results create a df ###
    results_df = pd.DataFrame({
    'Customer_Id': [customer_id],
    'Date': [today],
    'Time': [time],
    'Recency': [Recency],
    'Frequency': [Frequency],
    'Revenue': [Revenue],
    'Prediction': [pred]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x))


    st.table(results_df)

    if pred == 'Low Profile':
        st.image("https://static.thenounproject.com/png/2602104-200.png")

    elif pred == 'Medium Profile':
        st.image("https://static.thenounproject.com/png/2567989-200.png")

    elif pred == 'High Profile':
        st.image("https://static.thenounproject.com/png/2602211-200.png")    
else:
    st.markdown("Please click the *Submit Button*!")
