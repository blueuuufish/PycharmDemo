from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('detector.html')

@app.route('/send_text', methods=['POST'])
def send_text():
    data = request.json
    text = data['text']
    print(text)
    # 这里你可以处理文本，例如存储到数据库或进行其他操作
    return jsonify({"message": "Text received successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
