import os
import sys
sys.stdout = sys.stderr

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


#from application import app

#app = Flask(__name__)

#appli_context = application.app_context()
#appli_context.push()

if __name__ == '__main__':
    application.run(debug=False)
