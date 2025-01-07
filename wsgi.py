import os
import sys
sys.stdout = sys.stderr

from werkzeug.middleware.proxy_fix import ProxyFix

# generic path manipulation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, '/portal/web/cgi-bin/gmao/data-services/')
#sys.path.insert(0, '/explore/dataportal/applications/data-services-fluid/data_services/')
#sys.path.insert(0,'/explore/dataportal/applications/GMAO/fluiddev/fluid_dev/')
#sys.path.insert(0, '/portal/web/cgi-bin/gmao/data-services/data_services')
# solve issue with matplotlib imports
import matplotlib
matplotlib.use('Agg')

# main application
from scorecard_web import app as application

application.wsgi_app = ProxyFix(
    application.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

if __name__ == '__main__':
    application.run(debug=False)
