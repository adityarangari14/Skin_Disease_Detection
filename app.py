from flask import Flask, render_template,request
from mlcode import predict_skin_disease
from HAM100 import predict_cancer_class
import datetime


app = Flask(__name__)

i =1

@app.route('/predictcancer',methods=["GET"])
def index():
    return render_template('index2.html')

@app.route('/predictcancer',methods=['POST'])
def input2():
    img = request.files['image']
    current_datetime = datetime.datetime.now()
# Format the current date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m- %H:%M:%S")
    # imageName = formatted_datetime+"Disease_image.jpg"
    add = str(i)
    print(type(add))
    img.save(add+"2024-932901 Disease image.jpg")
    URL = add+"2024-932901 Disease image.jpg"
    print(URL)
    return  render_template('index2.html',disease_detected = predict_cancer_class(URL))


@app.route('/predict',methods=["GET"])
def index2():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def input():
    img = request.files['image']
    current_datetime = datetime.datetime.now()
# Format the current date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m- %H:%M:%S")
    # imageName = formatted_datetime+"Disease_image.jpg"
    add = str(i)
    print(type(add))
    img.save(add+"2024-932901 Disease image.jpg")
    URL = add+"2024-932901 Disease image.jpg"
    print(URL)
    return  render_template('index.html',disease_detected = predict_skin_disease(URL))

if __name__ == "__main__":
    app.run(debug=True)