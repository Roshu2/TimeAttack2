from flask import Flask, jsonify, render_template
from pymongo import MongoClient
import catdog

app = Flask(__name__)
client = MongoClient('localhost', 27017)
db = client.timeattack



@app.route('/')
def home():
    print("hello")

    return render_template('nohavetime.html')


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)