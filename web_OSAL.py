import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
#应用标题
st.set_page_config(page_title='Pred hypertension in  patients with OSA')
st.title('Prediction Model of Obstructive Sleep Apnea-related Hypertension: Machine Learning–Based Development and Interpretation Study')
st.sidebar.markdown('## Variables')
ESSL = st.sidebar.selectbox('ESSL',('Normal','Low','Middle','High'),index=1)
hypertension = st.sidebar.selectbox('hypertension',('No','Yes'),index=1)
BQL = st.sidebar.selectbox('BQL',('Low risk','High risk'),index=1)
SBSL = st.sidebar.selectbox('SBSL',('Low risk','High risk'),index=0)
drink = st.sidebar.selectbox('drink',('No','Yes'),index=1)
smork = st.sidebar.selectbox('smork',('No','Yes'),index=1)
snoring = st.sidebar.selectbox('snoring',('No','Yes'),index=1)
suffocate = st.sidebar.selectbox('suffocate',('No','Yes'),index=1)
memory = st.sidebar.selectbox('memory',('No','Yes'),index=1)
HSD = st.sidebar.selectbox('HSD',('No','Yes'),index=1)
HFD = st.sidebar.selectbox('HFD',('No','Yes'),index=1)
LOE = st.sidebar.selectbox('LOE',('No','Yes'),index=1)
gender = st.sidebar.selectbox('gender',('female','male'),index=1)
age = st.sidebar.slider("age(year)", 0, 99, value=45, step=1)
BMI = st.sidebar.slider("BMI", 15.0, 40.0, value=20.0, step=0.1)
waistline = st.sidebar.slider("waistline(cm)", 50.0, 150.0, value=100.0, step=1.0)
NC = st.sidebar.slider("NC(cm)", 20.0, 60.0, value=30.0, step=0.1)
DrT = st.sidebar.slider("DrT", 0, 50, value=30, step=1)
SmT = st.sidebar.slider("SmT", 0, 50, value=30, step=1)
SmA = st.sidebar.slider("SmA", 0, 5, value=3, step=1)
#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'No':0,'Yes':1,'Normal':0 ,'Low':1, 'Middle':2,'High':3, 'Low risk':0,'High risk':1,'female':0, 'male':1}
ESSL =map[ESSL]
hypertension = map[hypertension]
BQL = map[BQL]
SBSL =map[SBSL]
drink =map[drink]
smork = map[smork]
snoring = map[snoring]
suffocate = map[suffocate]
memory = map[memory]
HSD =map[HSD]
HFD =map[HFD]
LOE =map[LOE]
gender = map[gender]
# 数据读取，特征标注
hp_train = pd.read_csv('E:\\Spyder_2022.3.29\\data\\machinel\\lwl_data\\OSA\\serve_osa.csv')

hp_train['OSAL'] = hp_train['OSAL'].apply(lambda x : +1 if x==1 else 0)

features =["ESSL","hypertension","BQL","SBSL","drink",'smork',"snoring",'suffocate','memory','HSD','HFD','LOE','gender','age','BMI','waistline','NC',
             'DrT','SmT','SmA']
target = 'OSAL'
random_state_new = 50
# ros = RandomOverSampler(random_state=random_state_new, sampling_strategy='auto')
# X_ros, y_ros = ros.fit_resample(hp_train[features], hp_train[target])
# X_ros = np.array(X_ros)
# gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
# gbm.fit(X_ros, y_ros)
X_ros = np.array(hp_train[features])
y_ros = np.array(hp_train[target])
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
# XGB = XGBClassifier(n_estimators=360, max_depth=2, learning_rate=0.1,random_state = 0)
gbm.fit(X_ros, y_ros)


sp = 0.5
#figure
is_t = (gbm.predict_proba(np.array([[ESSL,hypertension,BQL,SBSL,drink,smork,snoring,suffocate,memory,HSD,HFD,LOE,gender,age,BMI,waistline,NC,
                                     DrT,SmT,SmA]]))[0][1])> sp
prob = (gbm.predict_proba(np.array([[ESSL,hypertension,BQL,SBSL,drink,smork,snoring,suffocate,memory,HSD,HFD,LOE,gender,age,BMI,waistline,NC,
                                     DrT,SmT,SmA]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for OSAL:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of High risk group:  '+str(prob)+'%')

