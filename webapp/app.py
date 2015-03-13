from flask import Flask, render_template, redirect, request
import cPickle as pickle
import jinja2
import sys
sys.path.insert(0, './model')
from cluster_model import Model
from bar_chart import barchart

app = Flask(__name__)

# HOME PAGE:
@app.route('/')
def homepage():
	return render_template('base.html')

@app.route('/input')
def index():
    return render_template('input.html')

@app.route('/output_data', methods=['GET','POST'])
def collect_and_output():
    pga_dict = {'1-Clear':'1', '2-Minimal':'2', '3-Mild':'3',
                '4-Moderate':'4', '5-Marked':'5', '6-Severe':'6'}
    data = request.form.to_dict()

    description = 'For a {0}, {1} patient that is {2} years old  whose ' \
                  'smoking status is {3} with  Psoriasis at {4} severity ' \
                  'and affects {5} percentage body surface, the following ' \
                  'are the most used treatments'.format(data['race'].lower(),
                   data['sex'].lower(), data['age'], data['smoking'].lower(), 
                   data['pga'],data['bsa'])
    data['pga']=pga_dict[data['pga']]
    
    #loading pickled model
    with open('model/model.pkl', 'rb') as fp:
        model = pickle.load(fp)

    #passing in user input to classify patient
    cluster = model.classify_new_patient(data)[0]
    url = model.cluster_results_dict[cluster]['bar_chart']
    plot_size = '.embed?width=860&height=640'
    barplot = url+plot_size
    return render_template('output.html', description=description, 
                           barplot = barplot)

# DEFINING PSORIASIS
@app.route('/info')
def definition():
    return render_template('psoriasis.html')

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8787, debug=True)