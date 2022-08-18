import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle
import pandas as pd


app = Flask(__name__)


#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")

@app.route('/about')
def about():
  
    return render_template("about.html")

@app.route('/teacher')
def teacher():
  
    return render_template("teacher.html")

@app.route('/contact')
def contact():
  
    return render_template("contact.html")

@app.route('/major_project')
def majorproject():
  
    return render_template("major_project.html")

@app.route('/decision')
def decision():
  
    return render_template("decision.html")

@app.route('/svm')
def svm():
  
    return render_template("svm.html")

@app.route('/randomforest')
def randomforest():
  
    return render_template("randomforest.html")

@app.route('/knnmodel')
def knnmodel():
  
    return render_template("knnmodel.html")

@app.route('/naivebayes')
def naivebayes():
  
    return render_template("naivebayes.html")

@app.route('/internship_models')
def models():
      
    return render_template("internship_models.html")

@app.route('/predict_major',methods=['GET'])
def predict1():
    
    
    '''
    For rendering results on HTML GUI
    '''
    area = float(request.args.get('area'))
    perimeter=float(request.args.get('perimeter'))
    major_axis=float(request.args.get('major_axis'))
    minor_axis=float(request.args.get('minor_axis'))
    eccentricity=float(request.args.get('eccentricity'))
    solidity=float(request.args.get('solidity'))
    convex_area=float(request.args.get('convex_area'))
    extent=float(request.args.get('extent'))
    roundness=float(request.args.get('roundness'))
    compactness=float(request.args.get('compactness'))
    model1=float(request.args.get('model1'))

    if model1==0:
      model=pickle.load(open('Major_Project/decision_model_major.pkl','rb'))
    elif model1==1:
      model=pickle.load(open('Major_Project/svm_major.pkl','rb'))
    elif model1==2:
      model=pickle.load(open('Major_Project/random_forest_major.pkl','rb'))
    elif model1==3:
      model=pickle.load(open('Major_Project/knn_major.pkl','rb'))
    elif model1==4:
      model=pickle.load(open('Major_Project/naive_major.pkl','rb'))
      

    dataset= pd.read_excel('Date_Fruit_Datasets.xlsx')
    X = dataset.iloc[:, [0,1,2,3,4,6,7,8,10,11]].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[area,perimeter,major_axis,minor_axis,eccentricity,solidity,convex_area,extent,roundness,compactness]]))
    if prediction==[0]:
      message="Date Fruit is BERHI type"
    elif prediction==[1]:
      message="Date Fruit is DEGLET type"
    elif prediction==[2]:
      message="Date Fruit is DOKOL type"
    elif prediction==[3]:
      message="Date Fruit is IRAQI type"
    elif prediction==[4]:
      message="Date Fruit is ROTANA type"
    elif prediction==[5]:
      message="Date Fruit is SAFAVI type"
    elif prediction==[6]:
      message="Date Fruit is SOGAY type"
    else:
      message="not a date fruit"
    
        
    return render_template('major_project.html', prediction_text='{}'.format(message))


@app.route('/predict_decision',methods=['GET'])
def predict2():
    
    
    '''
    For rendering results on HTML GUI
    '''
    credit=float(request.args.get('credit'))
    age=float(request.args.get('age'))
    tenure=float(request.args.get('tenure'))
    balance=float(request.args.get('balance'))
    products=float(request.args.get('products'))
    estimatedsalary=float(request.args.get('estimatedsalary'))
    geography=float(request.args.get('geography'))
    gender=float(request.args.get('gender'))
    active=float(request.args.get('active'))
    creditcard=float(request.args.get('creditcard'))
   

    
    model=pickle.load(open('project7_decision_model.pkl','rb'))
    
      

    dataset= pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,1] = labelencoder_X.fit_transform(X[:,1])
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,2] = labelencoder_X.fit_transform(X[:,2])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[credit,age,tenure,balance,products,estimatedsalary,geography,gender,active,creditcard]]))
    if prediction==0:
      message="Customer will Exited"
    else:
      message="Customer will Not exited"
    
        
    return render_template('decision.html', prediction_text='{}'.format(message))



@app.route('/predict_svm',methods=['GET'])
def predict3():
    
    
    '''
    For rendering results on HTML GUI
    '''
    credit=float(request.args.get('credit'))
    age=float(request.args.get('age'))
    tenure=float(request.args.get('tenure'))
    balance=float(request.args.get('balance'))
    products=float(request.args.get('products'))
    estimatedsalary=float(request.args.get('estimatedsalary'))
    geography=float(request.args.get('geography'))
    gender=float(request.args.get('gender'))
    active=float(request.args.get('active'))
    creditcard=float(request.args.get('creditcard'))
    

    
    model=pickle.load(open('project7_svm.pkl','rb'))
    
      

    dataset= pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,1] = labelencoder_X.fit_transform(X[:,1])
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,2] = labelencoder_X.fit_transform(X[:,2])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[credit,age,tenure,balance,products,estimatedsalary,geography,gender,active,creditcard]]))
    if prediction==0:
      message="Customer will Exited"
    else:
      message="Customer will Not exited"
    
        
    return render_template('svm.html', prediction_text='{}'.format(message))


  
@app.route('/predict_knn',methods=['GET'])
def predict4():
    
    
    '''
    For rendering results on HTML GUI
    '''
    Age = float(request.args.get('age'))
    SibSp=float(request.args.get('sibsp'))
    Parch=float(request.args.get('parch'))
    Fare=float(request.args.get('fare'))
    Gender=float(request.args.get('gender'))
    Pclass=float(request.args.get('pclass'))

    model = pickle.load(open('KNN_project4.pkl','rb')) 
    dataset= pd.read_csv('train.csv')
    X=dataset.iloc[:,[5,6,7,9,4,2]].values
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    prediction = model.predict(sc.transform([[Age,SibSp,Parch,Fare,Gender,Pclass]]))
    if prediction==0:

      message1="Passenger will not survive."
    
    else:
      message1="Passenger will survive."

    
    
        
    return render_template('knnmodel.html', prediction_text='KNN Model predicted : {}'.format(message1))



@app.route('/predict_randomforest',methods=['GET'])
def predict5():
    
    
    '''
    For rendering results on HTML GUI
    '''
    tenth = float(request.args.get('tenth'))
    twelth=float(request.args.get('twelth'))
    btech=float(request.args.get('btech'))
    sevsem=float(request.args.get('7sem'))
    sixsem=float(request.args.get('6sem'))
    fivesem=float(request.args.get('5sem'))
    final=float(request.args.get('final'))
    medium=float(request.args.get('medium'))
    
    
    model=pickle.load(open('project6_random_forest.pkl','rb'))
    
    dataset= pd.read_excel('DATASET education.xlsx')
    X = dataset.iloc[:, 0:8].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[tenth,twelth,btech,sevsem,sixsem,fivesem,final,medium]]))
    if prediction==0:
      message="Student not Placed"
    else:
      message="Student will be placed"
    
        
    return render_template('randomforest.html', prediction_text='Model  has predicted : {}'.format(message))



@app.route('/predict_naive',methods=['GET'])
def predict6():
    
    
    '''
    For rendering results on HTML GUI
    '''
    app = Flask(__name__)
    model = pickle.load(open('dataset1nlp.pkl','rb')) 
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    print(cv)
    corpus=pd.read_csv('corpus_dataset1.csv')
    corpus1=corpus['corpus'].tolist()
    X = cv.fit_transform(corpus1).toarray()
    text = request.args.get('text')
    text=[text]
    input_data = cv.transform(text).toarray()
    
    
    model = pickle.load(open('dataset1nlp.pkl','rb'))
    
    prediction = model.predict(input_data)
    if prediction==1:
      message="review is Positive"
    else:
      message="review is Negative"
    
        
    return render_template('naivebayes.html', prediction_text='Given Review is : {}'.format(message))

if __name__ == "__main__":
  app.run(debug=True)
  
