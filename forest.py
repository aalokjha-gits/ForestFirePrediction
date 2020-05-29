from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<string:page_name>')
def page(page_name):
    return render_template(page_name)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    error = None
    if request.method == 'POST':
        features = [int(x) for x in request.form.values()]
        final = [np.array(features)]
        prediction = model.predict_proba(final)
        output='{0:.{1}f}'.format(prediction[0][1], 2)
        if output>str(0.5):
            return render_template('index.html',pred=f'Your Forest is in Danger.\nProbability of fire occuring is {output}')
        else:
            return render_template('index.html',pred=f'Your Forest is Safe for now.\nProbability of fire occuring is {output}')


if __name__ == '__main__':
    app.run(debug=True)