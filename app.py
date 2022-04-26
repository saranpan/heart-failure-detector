#from data import import_dataset
from package.util import cat_num_feature_seperator, statistical_inference
import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from pandas_profiling import ProfileReport

import base64
import xlsxwriter
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

import requests
from urllib.request import urlopen
from pickle import load
from copy import deepcopy
from PIL import Image
from time import sleep
from datetime import date

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from streamlit_pandas_profiling import st_profile_report
from streamlit_chat import message as st_message


#import model # trained on shorturl.at/gyAR1
pickled_model = load(open('model.pkl', 'rb'))

#import the data (we once used)
#import_dataset('https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data')


# Streamlit page
st.set_page_config(page_title='Hobot',page_icon=':man:',
                    layout="wide"
     )

# Hide hamburget and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

@st.experimental_memo
def get_data(map=False,reverse_map=False):
    df = pd.read_csv('heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
    df = df.astype({"age": int, "platelets": int,"serum_creatinine":float})
    
    # mapped version
    if map:
        df.replace({'sex':{0:'Female',1:'Male'}} , inplace = True) 
        df.replace({0:False,1:True},inplace=True)
        df = df.astype({"serum_creatinine":float})

    return df

def reverse_data(df):
        rdf = deepcopy(df)
        rdf.replace({'sex':{'Female':0,'Male':1}} , inplace = True) 
        rdf.replace({False:0,True:1},inplace=True)
        return rdf

@st.experimental_memo
def get_meta_data():
    mdf = pd.read_csv('metadata.csv')
    return mdf

def html_reader(html_file):
    HtmlFile = open(html_file, 'r', encoding='utf-8')
    page = HtmlFile.read() 
    components.html(page,scrolling=False)

@st.experimental_singleton 
def load_model():
    pickled_model = load(open('model.pkl', 'rb'))

    return pickled_model

@st.experimental_singleton(show_spinner=False)
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.experimental_singleton 
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

def clear_old_cach():
    if 'call_button' in st.session_state:
                del st.session_state['call_button'] 

    if 'submit_yet' in st.session_state:
                del st.session_state['submit_yet']

#def styling_df(DEATH_EVENT):
#    colors = {
#        'True':'red',
#        'False':'green'
#        }
#    return f"background-color: {colors[DEATH_EVENT]}"
        


df = get_data()
mdf = get_meta_data()
query_df = df
#For page INSIGHT
map_df = get_data(map=True)
#For page PREDICTION
model = load_model()


lottie_dr = load_lottie_url('https://assets9.lottiefiles.com/packages/lf20_hqlvpwat.json')
lottie_dr2 = load_lottie_url('https://assets5.lottiefiles.com/packages/lf20_vPnn3K.json')
lottie_dr3 = load_lottie_url('https://assets6.lottiefiles.com/packages/lf20_cbajnb2e.json')
lottie_dr4 = load_lottie_url('https://assets4.lottiefiles.com/packages/lf20_8zle4p5u.json')
lottie_dr5 = load_lottie_url('https://assets5.lottiefiles.com/packages/lf20_nhp1heev.json')


img_tree_2 = Image.open(urlopen('https://github.com/wallik2/heart-failure-detector/blob/main/static/graph_tree/dtree_render%201.png?raw=true'))
img_tree_51 = Image.open(urlopen('https://github.com/wallik2/heart-failure-detector/blob/main/static/graph_tree/dtree_render%2050.png?raw=true'))
img_tree_89 = Image.open(urlopen('https://github.com/wallik2/heart-failure-detector/blob/main/static/graph_tree/dtree_render%2088.png?raw=true'))
img = [img_tree_2,img_tree_51,img_tree_89]
img_cfm = Image.open(urlopen('https://github.com/wallik2/heart-failure-detector/blob/main/static/confusion_matrix.png?raw=true'))

#img_brand = Image.open('static\Intro.png')







#---------side bar-------#

selected = option_menu(
        menu_title= None , 
        options=["Home", 'Insight','Prediction'],
        icons=['house', 'bar-chart','bi bi-speedometer2'], 
        menu_icon="cast",
        default_index = 0,
        orientation="horizontal")


if selected == 'Home':

    clear_old_cach()

    with st.container():
        #left_column , right_column = st.columns((1,0))
        #with left_column:
        st.subheader("Hi, I am Hobot :wave: :man: ")
        st.title("An A.I Robot who is cardiologists")
        st.write("I'm willing to assist a heart disease patients to prevent death, I was trained by Saran P.")
        st.write("[Check out my trainer linkedin profile >](https://www.linkedin.com/in/saran-pannasuriyaporn-1104071ab/)")
        
        #with right_column:
        #    st.image(img_brand,width=1500)
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
            html_reader('templates/my_estimator.html')
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
        st.write("[Check out the research paper >](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)")
        st.write("[Check out which data I was fed on >](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)")




# page INSIGHT

elif selected == 'Insight':  

    clear_old_cach()

    # Sidebar
    with st.sidebar:
        selected_type = option_menu(menu_title= "Explore" , 
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

    elif selected_type == 'Interesting insight':
        
        st.title("Welcome to Heart Museum")
        st.write("- We will let you explore on interesting insights on your own")
        st.write("---")
        st.header(":rice_scene: Gallery 1")
        st.caption("Sometime, all insight we really want is just the mean and standard deviation of the lab result")
        
        with st.container():
            
            left_column , _ , right_column = st.columns((3,1,3))
            
            with left_column:
                age = st.slider(label = 'Select a range of age',
                            min_value = 40, 
                            max_value = 95,
                            value = (40, 95),
                            step = 1)
                
                time = st.slider(label = 'Select a range of follow-up period',
                            min_value = 4, 
                            max_value = 285,
                            value = (4, 285),
                            step = 1)
            
   
            q_df = map_df.query("age >= @age[0] & age <= @age[1]")
            st.write("---")
            with right_column: 
                disc_feature =  st.selectbox('Which type of group do you want to compare',
                    ('anaemia','diabetes','high_blood_pressure','smoking','DEATH_EVENT'))


        gdf = q_df.groupby([disc_feature,'sex']).agg({'creatinine_phosphokinase':['mean','std'],
                                                    'ejection_fraction':['mean','std'],
                                                    'platelets':['mean','std'],
                                                    'serum_creatinine':['mean','std'],
                                                    'serum_sodium':['mean','std']}).round(2)
    

        with st.container():
            feature  = st.selectbox('Which lab result you want to know',
                ('creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium'))


            # Temporary obtain the mean & std of the selected feature
            ffM = gdf.loc[(False,'Female'),(feature,'mean')]
            tfM = gdf.loc[(True,'Female'),(feature,'mean')]

            ffS = gdf.loc[(False,'Female'),(feature,'std')]
            tfS = gdf.loc[(True,'Female'),(feature,'std')]


            fmM = gdf.loc[(False,'Male'),(feature,'mean')]
            tmM = gdf.loc[(True,'Male'),(feature,'mean')]

            fmS = gdf.loc[(False,'Male'),(feature,'std')]
            tmS = gdf.loc[(True,'Male'),(feature,'std')]      
            
            #Start making plots
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Female',
                x=['False','True'], y=[ffM,tfM],
                error_y=dict(type='data', array=[ffS, tfS])))
            fig.add_trace(go.Bar(
                name='Male',
                x=['False','True'], y=[fmM, tmM],
                error_y=dict(type='data', array=[fmS, tmS])))



            fig.update_layout(barmode='group',
                            title=f"Mean and Standard deviation of {feature}".replace('_',' ').title(),
                            xaxis_title=disc_feature,
                            legend_title="Sex",
                            font = {'size':16})

            fig.update_yaxes(rangemode="nonnegative")
            
            st.plotly_chart(fig)     
                
    elif selected_type == 'Inference':
        
        st.title("Statistical Inference :mag_right:")
        st.markdown("""
                 - Sometimes, we can learn something via statistics. We could derive the knowledge from our sample dataset
                 
                 - However, also note that the inference we got does not always true, for example, the proportion of who smoking and don't does not differ in DEATH_EVENT which is not true in real world.
                 So, Viewer discretion is required
                 """)
                       
        cat_df,num_df = cat_num_feature_seperator(map_df)
        cat_feat = cat_df.columns.tolist()
        
        st.write("---")
        st.header(":mortar_board: Labatory 1 : Two Proportion Inference ")
        
        with st.container():
            
            left_column , _ , right_column = st.columns((1,1,2))

            with left_column:
                cat_feat = cat_df.columns.tolist()
                selected_feat = cat_feat
                
                # temporary fixing for preserving the old selected label
                try:
                    ix = selected_feat.index(st.session_state.chosen2)
                    selected_feat = cat_feat[:ix] + cat_feat[ix+1:]
                    
                    ix2 = selected_feat.index(st.session_state.chosen)
                except:
                    ix2 = 0
                    
                feature_1 = st.selectbox('Between',
                                        selected_feat,   
                                        key='chosen',
                                        index=ix2)
                
                ix = cat_feat.index(st.session_state.chosen)
                selected_feat = cat_feat[:ix] + cat_feat[ix+1:]
                
                # temporary fixing for preserving the old selected label
                try:
                    ix2 = selected_feat.index(st.session_state.chosen2)
                except:
                    ix2 = 0
                
                ### Fix more : feature 2 does limit feature 1 
                feature_2 = st.selectbox('Is there any difference in',
                        selected_feat,   
                        key='chosen2',
                        index = ix2)
                

                significance = st.select_slider(
                    'Select a significance level',
                     options=[0.01, 0.025, 0.05, 0.1],
                     value = 0.05)

            
            with right_column:
                test = statistical_inference(map_df, feature_1, feature_2,significance=significance)
                st.plotly_chart(test.two_proportion_inference_plot())

        st.write(f"Your question : Is there any significantly difference of the proportion of {feature_2} between each level {feature_1} with {significance * 100}% significance level ?")
        if st.button("Proceed"):
            st.write("---")
            st.subheader("Step 1 : Hypothesis Testing")
            st.write(f"Null Hypothesis : There is no significantly difference of the proportion of {feature_2} between each level {feature_1}")
            st.write(f"Alternative Hypothesis : There is significantly difference of the proportion of {feature_2} between each level {feature_1}")
            
            st.subheader("Step 2 : Significance level")
            st.write(f"If the probability of true null hypothesis got rejected is lower than {significance}, it's acceptable to reject it")
            
            st.subheader("Step 3 : Statistical Testing")
            stat,p_value,conclusion = test.two_proportion_inference()
            
            st.write(f"Z-score : {stat}")
            st.write(f"P-value : {p_value}")
            
            if p_value > significance:
                st.warning(conclusion)
            else:
                st.success(conclusion)
                

        
        st.write("---")
        
        for i in range(10):
            st.text(" ") 
            
        
        st_lottie(lottie_dr5,height=600)
        st.text("                         New Inference Labatories are coming soon")
# page PREDICTION
elif selected == 'Prediction': 

    with st.container():
        st.title("Predict a patient")
        st.write("""
        We want to remind you once again to read our medical disclaimer before predicting. 
        Furthermore, Our model is suitable for only the patient who once had
        - :broken_heart: Heart failure 
        - :boom: Left ventricular systolic dysfunction

        For those who does not have the following properties may not obtain an accurate prediction
        """)

        if 'call_button' not in st.session_state:
                st.session_state['call_button'] = False
        
        proceed = st.checkbox("I understand and wish to proceed",key='proceed')
        proceed_2 = st.button("Call Hobot",disabled=not proceed)
        if proceed_2:
            st.session_state['call_button'] = True

    if proceed & st.session_state['call_button']:

        # Customize run time
        if 'submit_yet' not in st.session_state:
                st.session_state['submit_yet'] = 1

        def change_run_sec():
            st.session_state['submit_yet'] = 0


        run_sec = st.session_state['submit_yet']


        with st.spinner('We are directing chat to the Hobot'):
            sleep(run_sec)
        
        # Generate chat
        st.write("---")
        st.title("Chat Room")
        sleep(run_sec)
        st_message('Hello I am Hobot, an AI who is cardiologist',avatar_style='miniavs')
        sleep(run_sec)
        st_message("Hi",is_user=True)
        sleep(run_sec)
        st_message('To predict the chance of having heart failure',avatar_style='miniavs')
        sleep(run_sec)
        st_message('I would like to know your simple bio, Medical condition, and lab result',avatar_style='miniavs')
        sleep(run_sec)
        st_message("Here is mine",is_user=True)
        sleep(run_sec)


        with st.form(key='Form1'):
                st.caption("Now, filling out your Simple bio, Medical condition, and lab result respectively in this form")
                st.header("Step 1 : Your simple bio :bust_in_silhouette:")
                sex = st.selectbox(label = "What is your sex", 
                                                        options=map_df['sex'].unique())

                age = st.number_input(label = 'What is your age (years)',min_value=0,step=5,value=65)

                smoking = st.selectbox(label = "Do you smoke ?", 
                                    options=[False,True]
                                )  

                st.write("---")                        
                st.header("Step 2 : Your medical condition :pill:")

                anaemia = st.selectbox(label = "Do you have Anaemia ?", 
                                        options=map_df['anaemia'].unique()
                                            )
                high_blood_pressure = st.selectbox(label = "Do you have high blood pressure (>=130 mm/Hg)?", 
                                                options=[False,True]
                                            )
                diabetes = st.selectbox(label = "Are you Diabetes ?", 
                                                options=map_df['diabetes'].unique()    
                                            )
        
                st.write("---")
                st.header("Step 3 : Your Lab Result :page_with_curl:")

                creatinine_phosp = st.number_input(label = 'What is CPK enzyme level in your blood',
                                                    min_value=0 , step=100,value= 7350,
                                                    )
                
                platelets = st.number_input(label = 'How many platelets in your blood',
                                            min_value=0 , step=1000,value=800000,
                                            )   

                ejection_fraction = st.number_input(label = 'What is percentage of blood leaving',
                                            min_value=0 , step=1,value=52,
                                            )   

                serum_creatinine = st.number_input(label = 'What is creatinine level in your blood',
                                            min_value=0.00,value=5.35,
                                            )   

                serum_sodium = st.number_input(label = 'What is sodium level in your blood',
                                            min_value=0 , step=1,value=115,
                                            )   
                
                st.write("---")
                last_date = st.date_input(label = 'When did you received the last treatment from your doctor',max_value = date.today(),
                                        )
                time = date.today() - last_date
                time = time.days
                #st.write(f"You met them last {time.days} days")
                submit_button = st.form_submit_button("Send !",on_click=change_run_sec)

        if submit_button:
            input = {'age':age,'anaemia':anaemia,'creatinine_phosphokinase':creatinine_phosp,
                    'diabetes':diabetes,'ejection_fraction':ejection_fraction,'high_blood_pressure':high_blood_pressure,
                    'platelets':platelets,'serum_creatinine':serum_creatinine,'serum_sodium':serum_sodium,
                    'sex':sex,'smoking':smoking,'time':time}

            input = pd.DataFrame([input])

            input = reverse_data(input)
            
            output = model.predict(input)[0]
            output_prob = model.predict_proba(input)[0]

            st_message('Let me analyzing your information ... ',avatar_style='miniavs')
            sleep(3)
            if output == 0:
                sleep(1)
                st_message('🙂 Congratulation, you are more unlikely to have heart failure',
                            avatar_style='miniavs')
                
                st.balloons()
                st_message(f'However, If I was wrong. The chance of having heart failure is {round(output_prob[1],4)*100} %',
                            avatar_style='miniavs')    
            elif output == 1:
                st_message('☹️ We are sorry to hear that. With the following status, you are likely to have heart failure more than do not',
                            avatar_style='miniavs')

                st_message(f'The chance of having heart failure is {round(output_prob[1],4)*100} %',
                            avatar_style='miniavs')    

                sleep(2)
                st_message(f'We recommend you to take an extremely care of yourself and having somebody near you to help you when you have heart failure',
                            avatar_style='miniavs')   
            
            st.text(" ")
            left_column, _ , middle_column , _ , right_column = st.columns((4,2,2,1,4))
            
            with left_column:
                st.write("---")

            with middle_column:
                st.caption("end chat")

            with right_column:
                st.write("---")



            #st.write(output)
            #st.write(output_prob)


            

            #st.write(model.predict_proba(input))
        
