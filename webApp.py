from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image


app = Flask(__name__)

dic = {0 : 'Malam', 1 : 'Pagi'}


model = load_model('model.pkl')
#model._make_predict_function()

# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	img_path = request.form['my_image']
	i = image.load_img(img_path, target_size=(128,128))
	i = image.img_to_array(i)
	i = i.reshape(1, 128,128,3)
	p = model.predict(i)
	return dic[p[0]]


@app.route("/about")
def about_page():
	return "By : Pakar-If"
