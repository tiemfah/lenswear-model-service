from flask import Flask, Response

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def index():
    return Response(status=200)


@app.route("/predict", methods=["POST"])
def predict():
    response = {}
    matching_result = {
        'apparel_skirt_2116fd47-f9d1-4fa5-8f10-a2f667cdab81': 0.7,
        'apparel_skirt_52e53bc2-fa56-4e61-8828-e2a0633d09d1': 0.1,
        'apparel_skirt_c6510227-24a4-4479-8bc8-626396bdde50': 0.1,
        'apparel_skirt_d10eae55-3b35-4798-8a50-93f5905ade7b': 0.1
    }
    response['predictions'] = matching_result
    return response, 200


if __name__ == '__main__':
    app.run(debug=False, port=9000)
