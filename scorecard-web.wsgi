import os
import sys
sys.stdout = sys.stderr

# generic path manipulation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, '/explore/dataportal/applications/GMAO/fluiddev/fluid_dev/data_services')
# solve issue with matplotlib imports
import matplotlib
matplotlib.use('Agg')

# main application
from scorecard_web import app as application
