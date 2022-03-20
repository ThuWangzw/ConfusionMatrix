#! /usr/bin/python3
import os
import pickle
import json
import argparse
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template
from data.dataCtrler import dataCtrler
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/api/confusionMatrix', methods=["POST"])
def confusionMatrix():
    return jsonify(dataCtrler.getConfusionMatrix())

@app.route('/api/confusionMatrixCell', methods=["POST"])
def confusionMatrixCell():
    labels = request.json['labels']
    preds = request.json['preds']
    return jsonify(dataCtrler.getImagesInConsuionMatrixCell(labels, preds))

@app.route('/api/image', methods=["GET"])
def image():
    imageID = int(request.args['imageID'])
    return jsonify(dataCtrler.getImage(imageID))

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, default='/data/zhaowei/jittor-data/')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise Exception("The path does not exist.")

    evaluationPath = os.path.join(args.data_path, "evaluation.json")
    predictPath = os.path.join(args.data_path, "predict_info.pkl")
    trainImagePath = os.path.join(args.data_path, "trainImages.npy")
    
    with open(predictPath, 'rb') as f:
        predictData = pickle.load(f)
    with open(evaluationPath, 'r') as f:
        statisticData = json.load(f)
    trainImages = np.load(trainImagePath)
    
    dataCtrler.process(statisticData, predictData = predictData, trainImages = trainImages, reordered=True)

    app.run(port=args.port, host=args.host, threaded=True, debug=False)

if __name__ == "__main__":
    main()