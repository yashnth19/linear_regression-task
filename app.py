from flask import Flask, render_template, request,jsonify
# from flask_cors import CORS,cross_origin #falsk cors is used to run from different regions
import requests


import pandas as pd
# from pandas_profiling import ProfileReport
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV,ElasticNet,ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
app = Flask(__name__)

df =pd.read_csv("ai4i2020.csv")
y=df['Air temperature [K]']
x=df.drop(columns=['Air temperature [K]','UDI','Product ID','Type'])
cols=[i for i in x.columns]
scaler =StandardScaler()
arr=scaler.fit_transform(x)
lr=LinearRegression()
@app.route('/',methods=['GET'])  # route to display the home page
# @cross_origin() #not required for local deployment
def homePage():
    return render_template("index.html")

@app.route('/review',methods=['POST','GET'])
# @cross_origin() #for deploying accross origins
def index():
        return render_template('report.html')

@app.route('/multicollinearity_check',methods=['POST','GET'])
# @cross_origin()
def multicollinearity_check():
    vif_df=pd.DataFrame()
    vif_df['vif'] = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
    vif_df['feature'] = x.columns
    mydict=[]
    x_new = x
    for i in range(0,vif_df.shape[0]):
        mydict.append((vif_df.iloc[i,1],vif_df.iloc[i,0]))
        if vif_df.iloc[i,0]>=10:
            note=str(vif_df.iloc[i,1]) + "   :column will be dropped while building the model as VIF > 10"
            x_new.drop(columns=[vif_df.iloc[i,1]])
            mydict.append(note)
    # print (x_new)
    return render_template('results.html',mydict=mydict)

@app.route('/Lin_reg',methods=['POST','GET'])
# @cross_origin()
def Lin_reg():
    mydict2=[]
    if request.method=="POST":
        split_value=request.form.get('sp')
        print(split_value)
        seed_value = request.form.get('seed')
        print(seed_value)
        vif_df = pd.DataFrame()
        vif_df['vif'] = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
        vif_df['feature'] = x.columns
        x_new = x
        for i in range(0, vif_df.shape[0]):
            mydict2.append(("Feature,VIF :", vif_df.iloc[i, 1], vif_df.iloc[i, 0]))
            if vif_df.iloc[i, 0] >= 10:
                note = str(vif_df.iloc[i, 1]) + "   :column will be dropped while building the model as VIF > 10"
                x_new.drop(columns=[vif_df.iloc[i, 1]])
                mydict2.append(note)
        arr1 = scaler.fit_transform(x_new)
        x_train, x_test, y_train, y_test = train_test_split(arr1, y, test_size=float(split_value), random_state=int(seed_value))
        features= [i for i in x_new.columns]
        label= "Air temperature [K]"
        slopes=""


        if request.form.get('Lin_Reg_Type') == 'Without_Regularization':
            lr.fit(x_train, y_train)
            coeffs=[i for i in lr.coef_]
            intercept = lr.intercept_
            accuracy=str(lr.score(x_test,y_test))+" (Without_Regularization)"
        elif request.form.get('Lin_Reg_Type')=='Lasso':
            lassocv = LassoCV(cv=10, max_iter=200000, normalize=True)
            lassocv.fit(x_train, y_train)
            lasso = Lasso(alpha=lassocv.alpha_)
            lasso.fit(x_train, y_train)
            coeffs = [i for i in lasso.coef_]
            intercept = lasso.intercept_
            accuracy = str(lasso.score(x_test, y_test))+" (Lasso)"
        elif request.form.get('Lin_Reg_Type')=='Ridge':
            ridgecv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=10, normalize=True)
            ridgecv.fit(x_train, y_train)
            ridge = Ridge(alpha=ridgecv.alpha_)
            ridge.fit(x_train, y_train)
            coeffs = [i for i in ridge.coef_]
            intercept = ridge.intercept_
            accuracy = str(ridge.score(x_test,y_test)) + " (Ridge)"
        elif request.form.get('Lin_Reg_Type')=='Elastic_Net':
            elastic = ElasticNetCV(alphas=None, cv=10)
            elastic.fit(x_train, y_train)
            el = ElasticNet(alpha=elastic.alpha_, l1_ratio=elastic.l1_ratio_)
            el.fit(x_train, y_train)
            coeffs = [i for i in el.coef_]
            intercept = el.intercept_
            accuracy = str(el.score(x_test,y_test)) + " (Elastic_Net)"
        for i in range(0,len(features)):
            slopes= slopes + " " + str(coeffs[i])  + " * " + str(features[i]) + "  + "
        eqn = label + ' = ' + slopes + " + " + str(intercept)
        mydict2.append(eqn)
        mydict2.append("Accuracy of the model = " + accuracy)
    return render_template('results2.html',mydict2=mydict2)

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True)
