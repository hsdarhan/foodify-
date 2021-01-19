from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///model.db'
app.config ['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Storage(db.Model)
    numID = db.Column(db.Integer, primary_key = True)
    food = db.Column(db.String(20), primary_key = True)

    def __init__(self, numID, food):
        self.numID = numID
        self.food = food

@app.route('/')
def index():
    return "Foodify!"

@app.route('/upload', methods=['POST'])
def upload():
    pic = request.files['pic']
    if not pic:
        return 'No picture uploaded!', 400

    filename = secure_filename(pic.filename)

    if not filename:
        return 'Bad upload!', 400

    picture = pic.read()
    food = predict_image(picture)

    result = Storage(int(randint(0, 999999)), food)
    db.session.add(result)
    db.session.commit()

    return food

if __name__ == "__main__":
    app.run(debug=True)