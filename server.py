import numpy as np
from flask import Flask, render_template,request,jsonify,url_for,json
import pickle


model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
app = Flask(__name__, static_folder="static")
@app.route('/',methods=['POST','GET'])
def predict():
    if(request.method=="POST"):
        age=request.form.get("age")
        sex=request.form.get("sex")
        cp=request.form.get("chestPainType")
        trestbps=request.form.get("bloodPressure")
        chol=request.form.get("cholestrol")
        fbs=request.form.get("bloodSugar")
        restecg=request.form.get("ecg")
        thalach=request.form.get("maxHeartRate")
        exang=request.form.get("exerAgina")
        oldpeak=request.form.get("oldPeak")
        slope=request.form.get("st")
        ca=request.form.get("majorVessel")
        thal=request.form.get("thal")
        getData = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        getData = np.reshape(getData,(1,13))
        getData = sc.transform(getData)
        prediction = model.predict(getData)
        if(prediction[0]):
            # return "unhealthy"
            return render_template('unhealthy.html')
        else:
            # return "healthy"
            return render_template('healthy.html')

        return (str(prediction[0]))
    return render_template('heart_disease_analysis.html')

if __name__=='__main__':
    app.run(port=5000,host='0.0.0.0')

