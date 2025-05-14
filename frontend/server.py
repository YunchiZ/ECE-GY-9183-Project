from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import socket
import logging
import sys
import requests

app = Flask(__name__)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

API_URL = os.getenv("API_URL", "http://localhost:8080/predict")


@app.route("/")
def index():
    """提供主页"""
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return "404 - File Not Found", 404
    except Exception as e:
        logging.error(f"Error loading index page: {str(e)}")
        return f"500 - Server Error: {str(e)}", 500


@app.route("/<path:filename>")
def serve_static(filename):
    """提供静态文件"""
    try:
        return send_from_directory(".", filename)
    except FileNotFoundError:
        return "404 - File Not Found", 404


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()
        prediction_id = data.get("prediction_id")
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "no text"}), 400
        payload = {"prediction_id": prediction_id, "text": text}

        response = requests.post(API_URL, json=payload)

        if not response.ok:
            return jsonify({"error": f"API response error {response.status_code}"}), 500

        result = response.json()

        logging.info(f"Generated prediction: {result}")

        if prediction_id:
            result["prediction_id"] = prediction_id

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()

        prediction_id = data.get("prediction_id")
        cls_feedback = data.get("classification_feedback")
        id_feedback = data.get("identification_feedback")
        sum_feedback = data.get("summary_feedback")

        logging.info(f"Received feedback - ID: {prediction_id}")
        logging.info(f"Classification feedback: {cls_feedback}")
        logging.info(f"Identification feedback: {id_feedback}")
        logging.info(f"Summary feedback: {sum_feedback}")

        return jsonify(
            {"status": "success", "message": "Feedback submitted successfully"}
        )

    except Exception as e:
        logging.error(f"Error processing feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            print(f"Using default port: {port}")

    local_ip = get_local_ip()
    print(f"Starting server at http://{local_ip}:{port}")
    print(f"API endpoints:")
    print(f"- Predict: http://{local_ip}:{port}/predict")
    print(f"- Feedback: http://{local_ip}:{port}/feedback")

    app.run(host="0.0.0.0", port=port, debug=True)
