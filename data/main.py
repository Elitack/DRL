from flask import Flask, request
import json
app = Flask(__name__)


@app.route('/')
def hello():
    return '因子库'


@app.route('/query')
def query():
    jsonObj = {
        "sites": [
            {"name": "菜鸟教程", "url": "www.runoob.com"},
            {"name": "google", "url": "www.google.com"},
            {"name": "微博", "url": "www.weibo.com"}
        ]
    }
    tmp1 = json.dumps(jsonObj, ensure_ascii=False)
    tmp2 = json.loads(tmp1)
    searchword = request.args.get('key', 'GET方式传参')
    return tmp1

if __name__ == "__main__":
    app.run()
