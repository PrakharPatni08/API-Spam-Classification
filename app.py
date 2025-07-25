from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)
with open("spam.pkl","rb") as f:
    model = joblib.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ''
    if request.method == 'POST':
        msg = request.form.get('message','')
        pred = model.predict(([msg]))[0]
        result = "Spam ❌" if pred == 1 else "Not Spam ✅"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
