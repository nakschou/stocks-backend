from flask import Flask
app = Flask(__name__)

@app.route('/get_trending_stocks')
def get_trending_stocks():
    return 'Hello, World!'