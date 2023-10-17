from flask import Blueprint
from . import views
from . import v2

app = Blueprint(
                'scorecard',
                __name__,
                template_folder='templates',
                static_folder='../wxmaps/static',
                static_url_path='/static'
               )

def url(import_name, url_rules=[], **options):
    view = import_name
    for url_rule in url_rules:
        app.add_url_rule(url_rule, view_func=view, **options)

url(views.db_check_tool,['/db_check_tool/'],methods=['GET', 'POST'])
url(views.api,['/v1/'],methods=['GET', 'POST'])
url(v2.api_v2,['/v2/'],methods=['GET', 'POST'])
app.register_error_handler(404,v2.page_not_found)
app.register_error_handler(500,v2.page_not_found)
