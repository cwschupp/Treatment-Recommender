from flask import Flask, render_template, redirect, request
import ipdb
import jinja2

app = Flask(__name__)

# HOME PAGE:
@app.route('/')
def homepage():
	return render_template('base.html')

@app.route('/input')
def index():
    return render_template('input.html')

@app.route('/collect_data', methods=['POST'])
def collect():
	data = request.form.to_dict()

@app.route('/output')
def classifier():
	return render_template('output.html')

# DEFINING PSORIASIS
@app.route('/info')
def definition():
    return render_template('psoriasis.html')

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8787, debug=True)