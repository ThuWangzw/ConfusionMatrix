#! /usr/bin/python3
from dataclasses import dataclass
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


@app.route('/api/metadata', methods=["POST"])
def metaData():
    return jsonify(dataCtrler.getMetaData())

@app.route('/api/confusionMatrix', methods=["POST"])
def confusionMatrix():
    return jsonify(dataCtrler.getConfusionMatrix())

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--rawDataPath", type=str, default='../datasets/coco/')
    parser.add_argument("--bufferPath", type=str, default='buffer/')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    args = parser.parse_args()
    if not os.path.exists(args.rawDataPath):
        raise Exception("The path does not exist.")
    
    dataCtrler.process(args.rawDataPath, args.bufferPath, reordered=False)

    app.run(port=args.port, host=args.host, threaded=True, debug=False)

if __name__ == "__main__":
    main()