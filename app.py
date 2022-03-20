from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = model.predict(final_features)[0]
    return render_template('index.html', prediction_text='Customer Purchase Prediction is :{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)