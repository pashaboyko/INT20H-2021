#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
import os
import logg
from Model_Pipeline import Pipeline

log = None
app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'ВЕЛИКИЙ SYSAN делает вещи.'

if __name__ == "__main__":
    log_directory = 'log'

    pipeline = new Pipeline("filename.pkl")

    data = pipeline.pipelineData(data)

    log = logg.setup_logging('Server')
    log = logg.get_log("Web-server")
	app.run(debug=True,host='0.0.0.0')

