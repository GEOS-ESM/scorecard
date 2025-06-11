import os
import flask
from flask import Blueprint

app = Blueprint(
                'scorecard',
                __name__,
                template_folder='templates',
                static_folder='../wxmaps/static',
                static_url_path='/static'
               )

import views
