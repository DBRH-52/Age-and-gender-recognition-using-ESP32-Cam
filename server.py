# server.py
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
#from test_model import gender_model, age_model, gender_predictions, age_predictions

gender_model = load_model('Models/gender_model.h5')
#gender_model = tf.keras.models.load_model('Models/gender_model.h5')
#gender_model.save('Models/gender_model', save_format='tf')
#gender_model = tf.keras.models.load_model('Models/gender_model')
age_model = load_model('Models/age_model.h5')
#age_model = tf.keras.models.load_model('Models/age_model.h5')
#age_model.save('Models/age_model', save_format='tf')
#age_model = tf.keras.models.load_model('Models/age_model')


app = Flask(__name__)

def preprocess_image(image_array):
    image_resized = tf.image.resize(image_array, (224, 224))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']

    image_array = np.array(image_data)
    preprocessed_image = preprocess_image(image_array)

    gender_prediction = gender_model.predict(preprocessed_image)
    gender_result = 'male' if gender_prediction > 0.5 else 'female'
    age_prediction = np.argmax(age_model.predict(preprocessed_image), axis=1)
    age_result = age_prediction[0]

    return jsonify({
        'gender': gender_result,
        'age_group': age_result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #to chyba jednak nie ten port

