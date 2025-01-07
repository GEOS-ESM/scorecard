'''[GMAO FLUID]
Main Flask application that ties together multiple sub-applications.
'''
import os
import sys
import flask
import yaml

# filesystem
directory = os.path.dirname(__file__)
here = os.path.abspath(directory)
sys.path.append(here)

import scorecard

# main application
app = flask.Flask(__name__)

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    threaded=True

class ProdConfig(Config):
    FLASK_ENV = 'production'
    DEBUG = False
    TESTING = False

class DevConfig(Config):
    FLASK_ENV = 'development'
    DEBUG = True
    TESTING = True
    PREFERRED_URL_SCHEME='https'
    #SERVER_NAME='dphttpdev01.nccs.nasa.gov'
    #SERVER_NAME='fluid.dp.nccs.nasa.gov'
    #TRAP_HTTP_EXCEPTIONS = True
    #EXPLAIN_TEMPLATE_LOADING = True

app.config.from_object(Config)
#app.config.from_object(ProdConfig)
app.config.from_object(DevConfig) #Comment out on sync
app.jinja_env.add_extension('jinja2.ext.do')

# applications register
app.register_blueprint(scorecard.app)#, url_prefix='/scorecard')

# standard page titles
title = 'GMAO Scorecard Tool'
service = {
    'scorecard':'GEOS Scorecard',
}

def read_yml(file):
    with open(file, 'r') as ymlfile:
        file_dict=yaml.safe_load(ymlfile)
    return file_dict

@app.route("/robots.txt")
def robots():
    stat_dir = os.path.join(here, 'static')
    return flask.send_from_directory(directory=stat_dir, filename="robots.txt")

@app.route('/favicon.ico/')
def favicon():
     return flask.send_from_directory(os.path.join(app.root_path, 'static'),
        'img/nasa.ico', mimetype='image/vnc.microsoft.icon')

@app.route('/')
@app.route('/scorecard/')
def index():
    return flask.render_template('scorecard/landing.html')

@app.route('/about/')
def about():
    return flask.render_template(
        'about.html',
        service='About GMAO FLUID',
        title=title,
    )

if __name__ == '__main__':
    app.run(threaded=True, debug=False)
    #app.run(threaded=True)
