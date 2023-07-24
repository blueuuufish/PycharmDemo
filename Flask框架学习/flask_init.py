from flask import Flask

app = Flask(__name__)


@app.route("/")
def index():
    return "hello world!!!"


@app.route("/hi", methods=['POST'])
def hi():
    return "I am saying hi!"


app.run()
