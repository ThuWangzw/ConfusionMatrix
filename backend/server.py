#! /usr/bin/python3
from dataclasses import dataclass
import os
import pickle
import json
import argparse
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template, make_response
from data.dataCtrler import dataCtrler
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/api/metadata', methods=["POST"])
def metaData():
    return jsonify(dataCtrler.getMetaData())

@app.route('/api/confusionMatrix', methods=["POST"])
def confusionMatrix():
    query = None
    if 'query' in request.json:
        query = request.json['query']
    return jsonify(dataCtrler.getConfusionMatrix(query))


@app.route('/api/boxSizeDist', methods=["POST"])
def boxSizeDist():
    query = None
    if 'query' in request.json:
        query = request.json['query']
    return jsonify(dataCtrler.getBoxSizeDistribution(query))

@app.route('/api/boxAspectRatioDist', methods=["POST"])
def boxAspectRatioDist():
    query = None
    if 'query' in request.json:
        query = request.json['query']
    return jsonify(dataCtrler.getBoxAspectRatioDistribution(query))

@app.route('/api/image', methods=["GET"])
def imageGradient():
    boxID = int(request.args['boxID'])
    showmode = request.args['show']
    showall = request.args['showall']
    image_binary = dataCtrler.getImage(boxID, showmode, showall).getvalue()
    response = make_response(image_binary)
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set(
        'Content-Disposition', 'attachment', filename='%s.jpg' % boxID)
    return response

@app.route('/api/images', methods=["POST"])
def imagesGradient():
    boxIDs = request.json['boxIDs']
    showmode = request.json['show']
    return jsonify(dataCtrler.getImages(boxIDs, showmode))

@app.route('/api/imagesInCell', methods=["POST"])
def confusionMatrixCell():
    labels = request.json['labels']
    preds = request.json['preds']
    return jsonify(dataCtrler.getImagesInConsuionMatrixCell(labels, preds))

@app.route('/api/grid', methods=["POST"])
def grid():
    nodes = request.json['nodes']
    constraints = None
    if 'constraints' in request.json:
        constraints = request.json['constraints']
    depth = request.json['depth']
    aspectRatio = 1
    if 'aspectRatio' in request.json:
        aspectRatio = request.json['aspectRatio']
    return jsonify(dataCtrler.gridZoomIn(nodes, constraints, depth, aspectRatio))

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--rawDataPath", type=str, default='/data/yukai/UnifiedConfusionMatrix/datasets/coco/')
    parser.add_argument("--bufferPath", type=str, default='/data/zhaowei/ConfusionMatrix/buffer/')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    args = parser.parse_args()
    if not os.path.exists(args.rawDataPath):
        raise Exception("The path does not exist.")
    
    dataCtrler.process(args.rawDataPath, args.bufferPath, reordered=False)

    app.run(port=args.port, host=args.host, threaded=True, debug=False)

if __name__ == "__main__":
    main()