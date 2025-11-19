from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
#from builtins import *

'''
Notes:
- better performance
- printer friendly, downloadable view
- surface levels
- combine u and v winds
- column fix from Sterling
- test for missing data (missing date or missing field)
- click to obtain gmaopy difference and die-off curves
- ability to change forecast length
- verification - self, gfs, ecmwf
- RMS version
- put behind launchpad on production machine
- create API for ease of use for full scorecard
- use multiprocessing to handle multiple score retrievals and computations

Thought: use the template to call the method that retrieves scores and renders html as the response?

thoughts on printer friendly, downloadable view:
- can take score icons from html and just put them back toward a route through request
- use png creation like gpy?
- what is my template for the format? ECMWF

    future extensions:
        - output = 'compact' - printer/journal quality
        - input = ['obs', 'fcst']
https://http.cat/
https://httpstatusdogs.com/

- what happens when either or both exp/cntrl missing from request?
'''

import os
import sys
import json
import flask
import numpy as np
import datetime as dt
import itertools
from scipy.special import stdtr

import psycopg2
import logging
logging.basicConfig(level='INFO') # put in config?

from . import app

# app specific imports
directory = os.path.dirname(__file__)
here = os.path.abspath(directory)
sys.path.insert(0, here)

# contains functionality for doing database operations
import scorecard
import scorecard_util

@app.errorhandler(404)
def page_not_found(e):
    return '<img src="https://http.cat/404.jpg"><br><p>'+str(e.message)+'</p><br><p>'+str(e.description)+'</p>'

@app.errorhandler(500)
def page_not_found(e):
    return '<img src="https://http.cat/500.jpg"><br><p>'+str(e.message)+'</p><br><p>'+str(e.description)+'</p>'

@app.route('/v2/') #, methods=['GET', 'POST'])
def api_v2():
    '''changes from v1:
    - request method defaults to GET and removed POST option
    '''
    if flask.request.method != 'GET':
        # need to design custom error message pages
        # https://flask.palletsprojects.com/en/1.1.x/errorhandling/#generic-exception-handlers
        flask.abort(404, description='Sorry, HTTP request methods other than GET are unallowed currently.')

    # request input(s) handling
    if 'exp' not in flask.request.args:
        flask.abort(404, description='please supply an experiment name as argument in url as exp=experiment_name')
    if 'cntrl' not in flask.request.args:
        flask.abort(404, description='please supply a control experimental name as an argument in the url as cntrl=experiment_name')
    # need to test about dates, fcst_length, and other url parameters

    # inputs (assume they exist and are correct within database)
    args_exp = flask.request.args.get('exp', None)
    args_cntrl = flask.request.args.get('cntrl', None)

    # first, test db connection and find corresponding db
    try:
        db_exp, args_exp = scorecard_util.check_db(args_exp)
    except Exception as e:
        logging.error(e)
        flask.abort(500, description='experiment not found in database')
    try:
        db_cntrl, args_cntrl = scorecard_util.check_db(args_cntrl)
    except Exception as e:
        logging.error(e)
        flask.abort(500, description='control not found in database')

    logging.info('Exp:      ' + str(args_exp))
    logging.info('Exp DB:   ' + str(db_exp))
    logging.info('Cntrl:    ' + str(args_cntrl))
    logging.info('Cntrl DB: ' + str(db_cntrl))

    # need to validate dates (or use full database record instead)
    # need to check that experiment covers those dates
    # what about EC or GFS verifications?

    # DATES
    bdate = flask.request.args.get('bdate', None)
    edate = flask.request.args.get('edate', None)
    bdate = dt.date(int(bdate[:4]), int(bdate[4:6]), int(bdate[6:8]))
    edate = dt.date(int(edate[:4]), int(edate[4:6]), int(edate[6:8]))

    # pre-defined fields (need to change based on scorecard type)
    scorecard_type = flask.request.args.get('type', None) # defaults to regular scorecard
    if 'rms' in scorecard_type:
        stats = []
    else:
        stats = ['cor', 'rms']

    # stats = cor, rms, rms_bar, rms_dis, rms_dsp, rms_ran
    # fields = t, h, u, v, q, p, bc, du, oc, ss, t2m, tau, u10m, v10m
    fields = ['t', 'h', 'u', 'v', 'q', 'p'] # need to add surface fields
    domains = ['n.hem', 's.hem', 'tropics']

    return 'in progress'
