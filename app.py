import flask
import werkzeug
from keras.models import load_model
import numpy
import scipy.misc


app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    img = scipy.misc.imread(filename)
    img = scipy.misc.imresize(img, (320, 320))
    global model
    model = load_model('cnn77.model')
    result = model.predict_classes(img.reshape(1,320,320,3))
    if result == [0]:
        return "actinic keratosis"
    elif result == [1]:
        return "basal cell carcinoma"
    elif result == [2]:
        return "pigmented benign keratosis"
    elif result == [3]:
        return "dermatofibroma"
    elif result == [4]:
        return "melanoma"
    elif result == [5]:
        return "nevus"
    elif result == [6]:
        return "vascular lesion"







if __name__ == '__main__':

    app.run(host='0.0.0.0')



