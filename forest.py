from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import requests
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

# Weather API integration
url = "https://community-open-weather-map.p.rapidapi.com/weather"
headers = {
    'x-rapidapi-host': "community-open-weather-map.p.rapidapi.com",
    'x-rapidapi-key': "c710cba165msh6adf5ce1703d409p14d0ecjsn0770f2a59958"
    }
###########################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<string:page_name>')
def page(page_name):
    return render_template(page_name)

@app.route('/predict_by_city', methods=['POST', 'GET'])
def predict_by_city():
    error = None
    if request.method == 'POST':
        city_name = request.form['city_name']
        querystring = {"q": city_name}
        response = requests.request("GET", url, headers=headers, params=querystring)
        data = response.json()
        temp_in_kelvin = data['main']['feels_like']
        temp_in_deg_cel = temp_in_kelvin - 273.15
        humidity = data['main']['humidity']
    # return render_template('index.html',pred = f'{temperature}')
        # final = [np.array(features)]
        prediction = model.predict_proba([[temp_in_deg_cel,humidity]])
        output='{0:.{1}f}'.format(prediction[0][1], 2)
        if output>str(0.5):
            return render_template('index.html',pred=f'Your Forest is in Danger.\nProbability of fire occuring is {output}')
        else:
            return render_template('index.html',pred=f'Your Forest is Safe for now.\nProbability of fire occuring is {output}')

@app.route('/predict_by_geocordinate', methods=['POST', 'GET'])
def predict_by_geocordinate():
    error = None
    if request.method == 'POST':
        lat = request.form['latitude']
        lon = request.form['longitude']
        querystring = {"lat":lat,"lon":lon}
        response = requests.request("GET", url, headers=headers, params=querystring)
        data = response.json()
        temp_in_kelvin = data['main']['feels_like']
        temp_in_deg_cel = temp_in_kelvin - 273.15
        humidity = data['main']['humidity']
    # return render_template('index.html',pred = f'{temperature}')
        # final = [np.array(features)]
        prediction = model.predict_proba([[temp_in_deg_cel,humidity]])
        output='{0:.{1}f}'.format(prediction[0][1], 2)
        if output>str(0.5):
            return render_template('index.html',pred=f'Your Forest is in Danger.\nProbability of fire occuring is {output}')
        else:
            return render_template('index.html',pred=f'Your Forest is Safe for now.\nProbability of fire occuring is {output}')


if __name__ == '__main__':
    app.run(debug=True)
    