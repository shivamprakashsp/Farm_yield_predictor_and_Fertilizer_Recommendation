import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#Loading the dataset

yielddat=pd.read_csv('newnew.csv')
dfs = pd.read_excel('fert.xlsx')
dfs = dfs.drop('Fertilzers',axis=1)
# Scaling Yield dataset
scaler = StandardScaler() 
scaler.fit(yielddat.drop('Yield(Tonnes/Hectare)',axis=1))
scaled_features = scaler.transform(yielddat.drop('Yield(Tonnes/Hectare)',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=yielddat.columns[:-1])
X = df_feat
Y = yielddat['Yield(Tonnes/Hectare)']
#Training model for yield prediction
knn = KNeighborsRegressor(n_neighbors=11)
knn.fit(X,Y)
#Training model for fertilizer recommendation

X_train=dfs.drop('Class',axis=1)
y_train=dfs['Class']
knn1 = KNeighborsClassifier(n_neighbors=1)

knn1.fit(X_train,y_train)


st.write("""
# Crop Yield Prediction and Fertilizer Recommendation App
This app predicts the **Crop Yield** and recommends **Fertilizer** to be used on the basis of Weather, Soil Parameters and Desired Yield
""")

st.sidebar.subheader('Input for Yield Prediction')
#Input for Yield Prediction
def user_input_features_yield():
    temp = st.sidebar.slider('Average Temperature(C)', 0.000000, 35.000000, 17.191501)
    vpd = st.sidebar.slider('Vapour Pressure Deficit(kPa)', 0.000000,20.000000, 9.182233)
    precip = st.sidebar.slider('Precipitation(mm)', 0.000000,600.000000,99.146498)
    som = st.sidebar.slider('Soil Organic Matter(t/Ha)', 0.000000,15.000000,2.521281)
    awc = st.sidebar.slider('Available Water Capacity(fraction)', 0.000000, 0.300000, 0.166235)
    landar = st.sidebar.slider('Land Area(sq-m)', 40000.000000,7000000.000000,4.904763e+05)
    data1 = {'Temp': temp,'VPD': vpd,'Precipitation':precip,'SOM':som,'AWC':awc,'Land Area':landar}
    features1 = pd.DataFrame(data1, index=[0])
    st.header('User Input Parameters')
    st.subheader('Yield Prediction')
    st.write(features1)
    

    temp = (temp - scaler.mean_[0])/scaler.scale_[0]
    vpd  = (vpd -scaler.mean_[5]) /scaler.scale_[5]
    precip = (precip-scaler.mean_[1])/scaler.scale_[1]
    som=som-scaler.mean_[2]
    som=som/scaler.scale_[2]
    awc=awc-scaler.mean_[3]
    awc=awc/scaler.scale_[3]
    landar=landar-scaler.mean_[4]
    landar=landar/scaler.scale_[4]
    
    data = {'Temp': temp,'VPD': vpd,'Precipitation':precip,'SOM':som,'AWC':awc,'Land Area':landar}
    features = pd.DataFrame(data, index=[0])
    
    return features

#Input for Fertilizer Recomendation
def user_input_features_fert():
    fert = st.sidebar.slider('Desired Yield(t/Ha)', 75.0000, 350.000000, 170.000000)
    
    data = {'Desired Yield(t/Ha)': fert}
    features = pd.DataFrame(data, index=[0])
    
    return features


dfyield = user_input_features_yield()
st.sidebar.subheader('Input for Fertilizer Recomendation')
dffert = user_input_features_fert()



st.subheader('Fertilizer Recommendation')
st.write(dffert)

yieldpred= knn.predict(dfyield)

st.header('Result')
fertrec = knn1.predict(dffert)
st.subheader('Predicted Yield(t/Ha)')

st.write(yieldpred)
st.write("Accuracy-80.40%")
yieldpred=np.round_(yieldpred)
fig=plt.figure(figsize=[12.0,0.5])
axes=fig.add_axes([0,0,1,1])
axes.set_xlim([50,300])
axes.set_xlabel('Yield(t/Ha)')
axes.set_yticks([])
sns.distplot(yieldpred)
st.pyplot()


st.subheader('Recommended Fertilizer(N-P-K)')
if fertrec==1:
    st.write("0-0-0")
elif fertrec==2:
    st.write("44-15-17")
elif fertrec==3:
    st.write("46-15-25")
elif fertrec==4:
    st.write("69-15-25")
elif fertrec==5:
    st.write("69-30-40")
elif fertrec==6:
    st.write("80-15-40")
elif fertrec==7:
    st.write("80-30-0")
elif fertrec==8:
    st.write("80-30-25")
elif fertrec==9:
    st.write("80-30-40")
else:
    st.write("92-30-40")
st.write("Accuracy-78.00%")





