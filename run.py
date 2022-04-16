from data import import_dataset
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from pickle import load
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import base64
import plotly.express as px
from copy import deepcopy
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from time import sleep
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import xlsxwriter

#import model # trained on shorturl.at/gyAR1
pickled_model = load(open('model.pkl', 'rb'))

#import the data (we once used)
import_dataset('https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data')


# Streamlit
st.set_page_config(page_title='Hobot',page_icon=':man:',
                    layout="wide",
                    menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     })

@st.experimental_memo
def get_data(map=False):
    df = pd.read_csv('heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
    df = df.astype({"age": int, "platelets": int,"serum_creatinine":float})
    
    # mapped version
    if map:
        df.replace({'sex':{0:'Female',1:'Male'}} , inplace = True) 
        df.replace({0:False,1:True},inplace=True)
        df = df.astype({"serum_creatinine":float})

    return df

@st.experimental_memo
def get_meta_data():
    mdf = pd.read_csv('metadata.csv')
    return mdf

def html_reader(html_file):
    HtmlFile = open(html_file, 'r', encoding='utf-8')
    page = HtmlFile.read() 
    components.html(page,scrolling=False)
    

df = get_data()
mdf = get_meta_data()
query_df = df
#For page INSIGHT
map_df = get_data(map=True)

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

lottie_dr = load_lottie_url('https://assets9.lottiefiles.com/packages/lf20_hqlvpwat.json')
lottie_dr2 = load_lottie_url('https://assets5.lottiefiles.com/packages/lf20_vPnn3K.json')
lottie_dr3 = load_lottie_url('https://assets6.lottiefiles.com/packages/lf20_cbajnb2e.json')
lottie_dr4 = load_lottie_url('https://assets4.lottiefiles.com/packages/lf20_8zle4p5u.json')

img_tree_2 = Image.open('graph_tree\dtree_render 1.png')
img_tree_51 = Image.open('graph_tree\dtree_render 50.png')
img_tree_89 = Image.open('graph_tree\dtree_render 88.png')
img = [img_tree_2,img_tree_51,img_tree_89]
img_cfm = Image.open('graph_tree\confusion_matrix.png')







#---------side bar-------#

selected = option_menu(
        menu_title= None , 
        options=["Home", 'Insight','Prediction'],
        icons=['house', 'bar-chart','bi bi-speedometer2'], 
        menu_icon="cast",
        default_index = 0,
        orientation="horizontal")


if selected == 'Home':
    with st.container():
        st.subheader("Hi, I am Hobot :wave: :man: ")
        st.title("An A.I Robot who is cardiologists")
        st.write("I'm willing to assist a heart disease patients to prevent death, I was trained by Saran P.")
        st.write("[Check out my trainer linkedin profile >](https://www.linkedin.com/in/saran-pannasuriyaporn-1104071ab/)")
        
    with st.container():
        st.write("---")
        left_column , right_column = st.columns(2)
        with left_column:
            st.title("What we do")
            st.write("- We forecast if the patient died during the follow-up period with the following 12 characteristics of a heart disease patients")
            st.dataframe(mdf)
            st.write("- We will also let you see the insight of the patient who we have once tried to cure them")
        
        with right_column:
            st_lottie(lottie_dr,speed=1.3,height=600,quality="low")
        

    with st.container():
        st.write("---")
        left_column, _ , right_column = st.columns((4,1,8))
        with left_column:
            st_lottie(lottie_dr2,speed=1.3,height=400,quality="low")
        with right_column:
            st.header("Our Inspiration")
            st.markdown("""
            We known that Cardiovascular diseases (CVDs) are the number 1 cause of death globally. \n
            According to Policy Advice, around 17.9 million lives each year was taken from this disease, which accounts for 31% of all deaths worlwide. \n
            We all known that Heart failure is a common event caused by CVDs

            Fortunately, most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

            People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management. \n 
            Hence, we decided to implement a machine learning model in order to detect and prevent the people from being passed away by heart failure.
            
            
            """)

    with st.container():
        st.write("---")
        left_column, _ , right_column = st.columns((8,1,4))
        with left_column:
            st.header("Brief Information of our patients")
            st.markdown("""
            The data of patients was collected at the Faisalabad Institute of Cardiology \n and at the Allied Hospital in Faisalabad (Punjab, Pakistan), 
            during April–December 2015 [52, 66]. 
            The patients consisted of 105 women and 194 men, and their ages range between 40 and 95 years old. 
            All 299 patients had left ventricular systolic dysfunction and
            had previous heart failures that put them in classes III or IV of New York Heart Association (NYHA) classification of the stages of heart failure.
            For more information of our patients, check out the insight of data.
            """)    
        with right_column:
            st_lottie(lottie_dr3,speed=1.3,height=400,quality="low")

    with st.container():
        st.write("---")
        st.header("How we predict a patient")

        selectbox, left_column, right_column = st.columns((1,4,3))

        with right_column:
            
            st.write("- We used technique StandardScaler + RandomForestClassifier")
            html_reader('statics/my_estimator.html')
            st.write("- Standardizing the value helps prevent features with wider ranges from dominating the distance metric")
            st.write("- The Random Forest Classifier was chosen as our architecture, since its performance (f1 score) is better among others like Logistic Regression,Support Vector Classifier, and K-nearest neighbor ")
            
            st.write("- Our Random Forest Classifier comprises of 89 sub-trees, and use the average voting to obtain the prediction")
            st.write("- We choose three of our 89 subtrees, shown on the left-hand side")
        with selectbox:

            if 'index' not in st.session_state:
                st.session_state['index'] = 0

            select_1 = st.selectbox('Tree',['Tree 2', 'Tree 51', 'Tree 89'])

            if select_1 == 'Tree 2':
                st.session_state['index'] = 0
            elif select_1 == 'Tree 51':
                st.session_state['index'] = 1           
            elif select_1 == 'Tree 89':
                st.session_state['index'] = 2
                
        with left_column:
           st.image(img[st.session_state['index']],caption=(select_1))


    with st.container():
        st.write("---")
        st.header("Medical Disclaimer")    
        left_column, _ , right_column = st.columns((4,1,4))
        with left_column:
            st.write("""
            I am neither cardiologist nor doctor, all content and information on this website 
            is for informational and educational purpose only,
            and does not intend to substitute professional medical advice, diagnosis, or treatment.
            Always prioritize a doctor decision, 
            and never disregard professional medical advice or delay in seeking it because of something you have read on this website.
            """)

        with right_column:
            st_lottie(lottie_dr4,speed=1.3,height=300,quality="low")

    with st.container():
        st.write("---")
        st.header("Frequently Asked Questions (FAQ)")     
        with st.expander('How good this model could predict ?'):

            st.write("""
            There are many metrics to evaluate this model. We ignore metric Accuracy, since our data is imbalanced.
            The number of patients who have heart failure is double of who does not.
            When it comes to a medical stuff, we do not want the model to classify
            the patient who have high chance to have heart failure to be classified as safe.
            However, the tradeoff of concerning only recall would lead the model to be false alarm.
            Hence, we decided to use the metric with both concerning precision and recall. F1 Score
            On validation set, The F1 score is around 88%, while 75% on test set.
                    """)
            st.image(img_cfm,caption="Confusion matrix on a test set")
            st.write("""
                    According to confusion matrix, the model is more likely to make false prediction by
                    clasifying the people who is in danger as non-danger rather than non-danger to danger.
                    """)  

        with st.expander('Who is suitable for this model ?'):
            st.write("""
            Our model has many limitations. All patients that we trained on were all have previous heart failure(s), and
            also left ventricular systolic dysfunction, so we do expected this model to be used on the patient who once have those event.
            
            Furthermore, we do expect this model to be just an assistance of decision making for a doctor.

            WARNING : DO NOT SUBSTITUTE A DOCTOR BY THIS MODEL. USE THEM AS THEIR ASSISTANCE ONLY.""")

        with st.expander('Should I put trust in a prediction output of this model'):
            st.write("""

            Although we did balance on recall and precision in order to minimize the false prediction.
            However, we make no claims, promises or guarantees about the accuracy of the prediction output, and 
            we do not claim that this model is replacible by Cardiologist.
            If you (as a doctor or nurse) think that the prediction does not make any sense. Do not put trust in this model, go ahead and prioritize Doctor decision making.

                    """)  
       
    with st.container():
        st.write("---")
        st.header("Acknowledgement")
        st.write("Without the following data, we have no idea how to be Hobot, a cardiology experts")
        st.write("Thank you Davide Chicco & Giuseppe Jurman for publicly shared a research paper that analyzing 299 patients")
        st.write("Thank you LARXEL who upload the dataset in Kaggle")
        st.write("[Check out the research paper >](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)")
        st.write("[Check out which data I was fed on >](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)")




# page INSIGHT

if selected == 'Insight':  
    # Sidebar
    with st.sidebar:
        selected_type = option_menu(menu_title= "Type of Statistics" , 
                                options=["Overview of Data","Interesting insight", 'Inference'],
                                icons=['bi bi-server', 'bi bi-lightbulb','bi bi-graph-up-arrow'], 
                                menu_icon="cast",
                                default_index = 0,
                                orientation="vertical")
        



    if selected_type == 'Overview of Data':

        with st.sidebar:
            sex = st.multiselect(label = "Select sex of patient", 
                                            options=map_df['sex'].unique(),
                                            default=map_df['sex'].unique()
                                        )
            anaemia = st.multiselect(label = "Have Anaemia ?", 
                                            options=map_df['anaemia'].unique(),
                                            default=map_df['anaemia'].unique()
                                        )
            high_blood_pressure = st.multiselect(label = "Have high blood pressure ?", 
                                            options=map_df['high_blood_pressure'].unique(),
                                            default=[False,True]
                                        )
            diabetes = st.multiselect(label = "Have Diabetes ?", 
                                            options=map_df['diabetes'].unique(),
                                            default=map_df['diabetes'].unique()
                                        )
            smoking = st.multiselect(label = "Smoking ?", 
                                            options=map_df['smoking'].unique(),
                                            default=map_df['smoking'].unique()
                                        )               

            age = st.slider(label = 'Select a range of age',
                                min_value = 40, 
                                max_value = 95,
                                value = (40, 95),
                                step = 1)

            creatinine_phosp = st.slider(label = 'Select a range of the CPK enzyme level in the blood',
                                        min_value = 20, 
                                        max_value = 7890,
                                        value = (23, 7861),
                                        step = 100)
        
            platelets = st.slider(label = 'Select a range of the platelets in the blood',
                                        min_value = 25100, 
                                        max_value = 850000,
                                        value = (25100, 850000),
                                        step = 1)

            ejection_fraction = st.slider(label = 'Select a range of Percentage of blood leaving',
                                        min_value = 14, 
                                        max_value = 80,
                                        value = (14, 80),
                                        step = 1)

            serum_creatinine = st.slider(label = 'Select a range of creatinine level in the blood',
                                        min_value = 0.5, 
                                        max_value = 9.4,
                                        value = (0.5, 9.4),
                                        step = 0.1)

            serum_sodium = st.slider(label = 'Select a range of sodium level in the blood',
                                        min_value = 113, 
                                        max_value = 148,
                                        value = (113, 148),
                                        step = 1)
            
            time = st.slider(label = 'Select a range of follow-up period',
                                        min_value = 4, 
                                        max_value = 285,
                                        value = (4, 285),
                                        step = 1)
        
            query_df = map_df.query(
                                    '''
                                        sex == @sex & \
                                        anaemia == @anaemia & \
                                        high_blood_pressure == @high_blood_pressure & \
                                        diabetes == @diabetes & \
                                        smoking == @smoking & age >= @age[0] & age <= @age[1] & \
                                        creatinine_phosphokinase >= @creatinine_phosp[0] & creatinine_phosphokinase <= @creatinine_phosp[1] & \
                                        platelets >= @platelets[0] & platelets <= @platelets[1] & \
                                        ejection_fraction >= @ejection_fraction[0] & ejection_fraction <= @ejection_fraction[1] & \
                                        serum_creatinine >= @serum_creatinine[0] & serum_creatinine <= @serum_creatinine[1] & \
                                        serum_sodium >= @serum_sodium[0] & serum_sodium <= @serum_sodium[1] & \
                                        time >= @time[0] & time <= @time[1]
                                    ''')

        st.info("Tips : You may customize filter of data at the left sidebar")
        st.title("Overview of Data :open_file_folder:")
        
        with st.container():
            st.write("- This section explores Exploratory data analysis (EDA) of the patients")
            st.write("- Currently, there are two available options; Preview data and Generate a Report ")
            st.write("""
            Preview Data is to simply display the data of our patients. Perhap you do not want to 
            view the whole dataframe, but rather to see the filtered data that you did at the left sidebar. then you can
            adjust the filter while you preview data as well. We also provide you two ways to download your filtered dataset
            as csv and excel file.
            If you need a deeper analysis, then it's worth trying 
            to generate a report (for your filtered data) even it may take a half minute to generate.

            Good thing about the report you generated is that you can see the interaction of any pair of features.
            """)

        # Default ticking the checkbox
        with st.container():
            if 'preview' not in st.session_state:
                    st.session_state['preview'] = True
            
            left_column, _ ,right_column = st.columns((14,1,1))

            with left_column: 
                check_box = st.checkbox("Preview data",key='preview')

                if check_box:
                    st.dataframe(query_df)   

            with right_column:
                if check_box:
                    st.download_button(label="📥 Download csv File",
                                        data=query_df.to_csv(),
                                        file_name='query_dataset.csv')  

                    st.download_button(label="📥 Download Excel File",
                                        data=to_excel(query_df),
                                        file_name='query_dataset.xlsx')

        with st.container():
            button_g = st.button("Generate a Report")        

            if button_g:
                report = ProfileReport(query_df,title="An overview of 299 patients",explorative=False,
                                        missing_diagrams={'bar':False,'matrix':False,'heatmap':False,'dendrogram':False},
                                        samples={"head": 0, "tail": 0})
                st_profile_report(report)
                sleep(1)
                st.balloons()

        
# page PREDICTION
if selected == 'Prediction': 
    st.title("predict")


#---------side bar-------#
