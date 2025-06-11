from __future__ import print_function

import logging
logging.basicConfig(level='WARN')

import os
import flask
import datetime as dt
import json
import numpy as np
import itertools
from scipy.special import stdtr

import sys
import psycopg2

if __name__ != '__main__':
   from . import app

import scorecard
import argparse

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib
#matplotlib.use('Agg')
from PIL import Image, ImageChops
import datetime

##############################################################################
def parse_args(args=None):
    '''command-line argument parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp', help='experiment name'
    )
    parser.add_argument(
        '--cntrl', help='control name'
    )
    parser.add_argument(
        '--bdate', help='beginning date; YYYYMMDD'
    )
    parser.add_argument(
        '--edate', help='ending date; YYYYMMDD'
    )

    if not args:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    try:
        args = parser.parse_args(args)
    except:
        if len(args) <= 1:
            parser.print_usage(sys.stderr)
            sys.exit(1)
        else:
            parser.print_usage(sys.stderr)
            sys.exit(2)

    return args
##############################################################################

def check(db=None, name=''):
    if db:
        con = scorecard.connection.Connection(db=db)
        if name:
            if con.check(name):
                logging.info('Successfully located {0} in {1} database.'.format(name, db))
                return name
            elif con.check(name + '.21z'):
                name += '.21z'
                logging.info('Successfully located {0} in {1} database.'.format(name, db))
                return name
            else:
                logging.warning('Experiment {0} not found in database {1}.'.format(name, db))
                return None
        else:
            logging.error('No experiment name specified.')
            return name
    else:
        logging.error('No database specified.')
        return db

def check_db(exp):
    # prioritize operational database - 20190130 (someone entered FPP data into exp database)
    o = check(sorted(scorecard.config.db.keys())[1], exp)
    if o:
        return (sorted(scorecard.config.db.keys())[1], o)
    else:
        e = check(sorted(scorecard.config.db.keys())[0], exp)
        if e:
            return (sorted(scorecard.config.db.keys())[0], e)
        else:
            logging.error('Experiment {0} not found in any database.'.format(exp))
            return None

def check_db_lev(exp_con, exp, fields, domains, stats):
    #print(exp_con.get(exp, num=1, **{'variable':fields[0], 'domain_name':domains[0], 'statistic':stats[0]}))
    results_70 = exp_con.get(exp, num=1, **{'variable':fields[0], 'domain_name':domains[0], 'statistic':stats[0], 'level':70})
    results_10 = exp_con.get(exp, num=1, **{'variable':fields[0], 'domain_name':domains[0], 'statistic':stats[0], 'level':10})
    return results_70 + results_10

# from gmaopy/stats/critval.py
# look at gmaopy/score/diffplot.py for reference
def critval(confidence, size):
    "It calculates the value of a t-student with (size-1) degrees of freedom corresponding to       \
     a given confidence interval. (ie. it returns t such as P(-t<T<=t)=confidence where t~t_(n-1))  \
     The t-value is computed by using the bisection method. "
    if confidence>1:
        raise Exception("Confidence has to be between zero and one.")
    # upper bound of integral [-inf, thigh]
    thigh=1.0
    # midpoint calculation from bisection method (difference is confidence height for upper)
    chigh=stdtr(size-1,thigh)-stdtr(size-1,-thigh)
    # raise the upper confidence level?
    while chigh<confidence:
        # i
        thigh*=2
        # midpoint calculation from bisection method (difference is confidence height for upper)
        chigh=stdtr(size-1,thigh)-stdtr(size-1,-thigh)
    # lower bound of zero
    tlow=0
    # if tlow is zero, then this is just 1/2 thigh (bad programming!)
    tcrit=0.5*(thigh+tlow)
    # why loop through 20 times? - bisection method N limit to not go to inf
    for i in range(20):
        # this is the same as the last chigh calculation (on first iteration)
        # subsequent iterations are supposed to narrow 
        c=stdtr(size-1,tcrit)-stdtr(size-1,-tcrit)
        # raise tcrit if chigh(or c in this case) is lower than 90%
        #print(c, confidence, tcrit)
        # raise lower bound up
        if c<confidence:
            tlow=tcrit
        # lower higher bound down
        else:
            thigh=tcrit
        # take midpoint
        tcrit=0.5*(thigh+tlow)
    return tcrit

class Identical(object):
    def transform(self, value):
        return value

    def transform_back(self, value):
        return value

    def difference(self, control, experiment):
        # includes a factor of -1 due to legend on website used
        # for both rms and cor which positive differences for
        # either are respectively bad/good
        return control - experiment

    def significance(self,reference,exp, lev=0.95):
        '''
        diff: difference between reference and experiment
            note: RMS is negative of COR
        critvals: t-student value for 90% confidence level for each n?
        '''

        # Will's doesn't work for RMS significances
        if self.name() in 'cor':
            diff = self.difference(reference, exp)
    
            # modified calc from Will in regards to difference in transform means
            zreference = self.transform(reference)
            zexp = self.transform(exp)
    
            # take mean while transformed in z-space
            ztmn1 = np.mean(zreference, 0)
            ztmn2 = np.mean(zexp, 0)
    
            # transform back to normal space
            referencemn = self.transform_back(ztmn1)
            expmn = self.transform_back(ztmn2)
    
            # 0.5 * diff = half the difference because we are using means?
            ztdiff = 0.5 * np.log( (1.0 + 0.5*diff)  / (1.0 - 0.5*diff) ) # why is there no additio of 1e-6?
    
            ztmn = np.mean(ztdiff, 0)
            ztvar = np.var(ztdiff, 0)
    
            dof = np.ma.count(diff, 0) # changed due to GMAOPy structure of arrays
            crits = np.zeros(dof.shape[0])
            for i in range(len(dof)):
                crits[i] = critval(lev, dof[i])
            zcrit = crits * np.sqrt(ztvar / dof)
            cordiff = self.difference(referencemn, expmn)
            corup   =  2 * ( (np.exp(2 * zcrit) - 1)  / (np.exp(2 * zcrit) + 1) ) # why 2?
            corlow  =  2 * ( (np.exp(-2 * zcrit) - 1)  / (np.exp(-2 * zcrit) + 1) )

            return cordiff, corlow, corup

        else:

            diff = self.difference(reference,exp)
            v = self.transform(diff)
            n = np.ma.count(v,0)
            v = v - np.mean(v,0)
            w = (np.sum(v * v,0)) / (n - 1)
            critvals = np.zeros(n.shape[0])
            for i in range(len(n)):
                critvals[i] = critval(lev,n[i])
            dx = critvals * np.sqrt(w / n)
            upper = self.transform_back(dx)
            lower = -upper
            diff = self.transform_back(diff)
    
            #print(lev)
            #print(cordiff)
            #print(corlow)
            #print(lev)
            #print(self.mean(diff))
            #print(lower)
            return diff,lower,upper

class Correlation(Identical):
    def transform(self, value):
        # transforms to space
        return 0.5 * np.log((1.0 + value) / (1.0 - value + 5.0e-6))

    def mean(self,value):
        null = value == 0
        #print('mean', value)
        #print('mean', value)
        #print('type: ', type(value))
        if not np.any(null):
            #print(self.transform)
            transform = self.transform(value)
            m = np.mean(transform,0)
            m = self.transform_back(m)
        else:
            m = np.mean(value,0)
        count = np.ma.count(value,0)
        if isinstance(count,np.ndarray):
            count = count[0]
        #print('mean calculated: ', m)
        #print('type: ', type(value))
        return m

    def transform_back(self, value):
        return (np.exp(2 * value) - 1) / (np.exp(2 * value) + 1)

    def difference(self, control, experiment):
        return experiment - control

    def name(self):
        return 'cor'

class RootMeanSquare(Identical):

    def name(self):
        return 'rms'

    def mean(self,value):
        #v = value * value
        #print('mean', value)
        m = np.mean(value, 0)
        #m = np.sqrt(m)
        count = np.ma.count(value,0)
        if isinstance(count,np.ndarray):
            count = count[0]
        return m

#@app.route('/v2/', methods=['GET', 'POST'])
#def api_v2():
#    return 'in progress'

##############################################################################
def main(arguments=[]):
    args = parse_args(arguments)
    if arguments:
        api_cli(args.exp, args.cntrl, args.bdate, args.edate)
    return 1

def api_cli(args_exp, args_cntrl, bdate, edate):
    '''CLI to api'''
    # first, test db connection and find corresponding db
    try:
        db_exp, args_exp = check_db(args_exp)
    except Exception as e:
        logging.error(e)
        #flask.abort(500)
        sys.exit(2)
    try:
        db_cntrl, args_cntrl = check_db(args_cntrl)
    except Exception as e:
        logging.error(e)
        #flask.abort(500)
        sys.exit(2)
    print(db_exp, args_exp, db_cntrl, args_cntrl)

    # create db connections
    exp = scorecard.connection.Connection(db=db_exp)
    cntrl = scorecard.connection.Connection(db=db_cntrl)

    # get dates/times
    bdate_code = bdate
    bdate = dt.date(int(bdate[:4]), int(bdate[4:6]), int(bdate[6:8]))
    edate_code = edate
    edate = dt.date(int(edate[:4]), int(edate[4:6]), int(edate[6:8]))

    card = do_work(db_exp, args_exp, exp, args_cntrl, cntrl, bdate, edate)

    # with an api, we should return only a single score block as a json object and let other routes use the api in
    # conjunction with forms and user interactivity
    verify = 'self'
    if '.21z' in args_exp:
        args_exp = args_exp[:-4]
    if '.21z' in args_cntrl:
        args_cntrl = args_cntrl[:-4]
    if '_ec' in args_exp:
        args_exp = args_exp[:-3]
        verify = 'ecwmf'
    if '_ec' in args_cntrl:
        args_cntrl = args_cntrl[:-3]
        verify = 'ecmwf'
    # [args_exp if 'x0035' not in args_exp else 'x0037_noSPPT'][0]
    bdate = bdate.strftime('%B %d, %Y').replace(' 0', ' ')
    edate = edate.strftime('%B %d, %Y').replace(' 0', ' ')
    #return jinja2.render_template(
    #        'scorecard/landing.html', data=json.loads(json.dumps(card, indent=4, sort_keys=True, separators=(',', ': '))),
    #        service="GEOS Scorecard", exp=args_exp, bdate=bdate, edate=edate, cntrl=args_cntrl, verify=verify,
    #    )

    scorecard_img(card, args_exp, args_cntrl, bdate, edate, bdate_code, edate_code)

    ##
    ## NOTE: THIS IS WHERE THE PNG IS RENDERED FROM THE card DICTIONARY
    ##

def scorecard_img(card, args_exp, args_cntrl, bdate, edate, bdate_code, edate_code):

    f_format = 'PNG'

    make_card('n.hem', card)
    print('Northern Hem Generated')
    make_card('s.hem', card)
    print('Southern Hem Generated')
    make_card('tropics', card)
    print('Tropics Generated')

    compile_card(args_exp, args_cntrl, bdate, edate, bdate_code, edate_code, f_format)
    print('Scorecard Generated')

    # Remove the regional card png files
    for f in ['n.hem','s.hem','tropics']:
        os.remove('{}_card_cli_temp.png'.format(f))

def get_syms(data):
    u_tri = u'\u25b2'
    uo_tri = u'\u25b3'
    block = u'\u2588'
    fuzz = u'\u2591'
    do_tri = u'\u25bD'
    d_tri = u'\u25bC'

    dat_str = ''

    if data == 3:
        dat_str += u_tri
        color = 'g'
    elif data == 2:
        dat_str += uo_tri
        color = 'g'
    elif data == 1:
        dat_str += fuzz
        color = 'g'
    elif data == 0:
        dat_str += block
        color = (0.7,0.7,0.7)
    elif data == -1:
        dat_str += fuzz
        color = 'r'
    elif data == -2:
        dat_str += do_tri
        color = 'r'
    elif data == -3:
        dat_str += d_tri
        color = 'r'

    return [dat_str, color]


def make_card(region, data):
    data = data[region]
    region_dict = {'n.hem': 'Northern Hemisphere', 's.hem': 'Southern Hemisphere', 'tropics': 'Tropics'}
    region_key = region
    region = region_dict[region_key]

    cell_text = []

    n_rows = len(data)
    columns = ['Variable', 'Pressure\nLevel', 'COR', 'RMS']
    rows = ['Geopotential\nHeight', 'SLP', 'Specific\nHumidity', 'Temperature', 'U-Wind', 'V-Wind']
    y_offset = np.zeros(len(columns))

    g_rows = 45
    g_cols = 11

    card = plt.figure(figsize = (6, 10), constrained_layout=False)
    gs = gridspec.GridSpec(ncols=g_cols, nrows=g_rows, figure=card)
    ax1 = card.add_subplot(gs[:2, 0:])
    ax2 = card.add_subplot(gs[2:5, 0:3])
    ax3 = card.add_subplot(gs[5:6, 0:5])
    ax4 = card.add_subplot(gs[6:13, 0:3])

    ax5 = card.add_subplot(gs[13:14, 0:3])
    ax6 = card.add_subplot(gs[14:21, 0:3])
    ax7 = card.add_subplot(gs[21:28, 0:3])
    ax8 = card.add_subplot(gs[28:35, 0:3])
    ax9 = card.add_subplot(gs[35:42, 0:3])

    ax10 = card.add_subplot(gs[2:5, 3:5]) # Pressure Level
    ax11 = card.add_subplot(gs[6:7, 3:5]) # 10
    ax12 = card.add_subplot(gs[7:8, 3:5]) # 70
    ax13 = card.add_subplot(gs[8:9, 3:5]) # 100
    ax14 = card.add_subplot(gs[9:10, 3:5]) # 250
    ax15 = card.add_subplot(gs[10:11, 3:5]) # 500
    ax16 = card.add_subplot(gs[11:12, 3:5]) # 700
    ax17 = card.add_subplot(gs[12:13, 3:5]) # 850
    ax18 = card.add_subplot(gs[13:14, 3:5]) # 1000

    ax19 = card.add_subplot(gs[14:15, 3:5]) # 10 sh
    ax20 = card.add_subplot(gs[15:16, 3:5]) # 70 sh
    ax21 = card.add_subplot(gs[16:17, 3:5]) # 100 sh
    ax22 = card.add_subplot(gs[17:18, 3:5]) # 250 sh
    ax23 = card.add_subplot(gs[18:19, 3:5]) # 500 sh
    ax24 = card.add_subplot(gs[19:20, 3:5]) # 700 sh
    ax25 = card.add_subplot(gs[20:21, 3:5]) # 850 sh

    ax26 = card.add_subplot(gs[21:22, 3:5])
    ax27 = card.add_subplot(gs[22:23, 3:5])
    ax28 = card.add_subplot(gs[23:24, 3:5])
    ax29 = card.add_subplot(gs[24:25, 3:5])
    ax30 = card.add_subplot(gs[25:26, 3:5])
    ax31 = card.add_subplot(gs[26:27, 3:5])
    ax32 = card.add_subplot(gs[27:28, 3:5])

    ax33 = card.add_subplot(gs[28:29, 3:5])
    ax34 = card.add_subplot(gs[29:30, 3:5])
    ax35 = card.add_subplot(gs[30:31, 3:5])
    ax36 = card.add_subplot(gs[31:32, 3:5])
    ax37 = card.add_subplot(gs[32:33, 3:5])
    ax38 = card.add_subplot(gs[33:34, 3:5])
    ax39 = card.add_subplot(gs[34:35, 3:5])

    ax40 = card.add_subplot(gs[35:36, 3:5])
    ax41 = card.add_subplot(gs[36:37, 3:5])
    ax42 = card.add_subplot(gs[37:38, 3:5])
    ax43 = card.add_subplot(gs[38:39, 3:5])
    ax44 = card.add_subplot(gs[39:40, 3:5])
    ax45 = card.add_subplot(gs[40:41, 3:5])
    ax46 = card.add_subplot(gs[41:42, 3:5])

    ax47 = card.add_subplot(gs[2:5, 5:8])
    ax48 = card.add_subplot(gs[5:6, 5:8])
    ax49 = card.add_subplot(gs[6:7, 5:8])
    ax50 = card.add_subplot(gs[7:8, 5:8])
    ax51 = card.add_subplot(gs[8:9, 5:8])
    ax52 = card.add_subplot(gs[9:10, 5:8])
    ax53 = card.add_subplot(gs[10:11, 5:8])
    ax54 = card.add_subplot(gs[11:12, 5:8])
    ax55 = card.add_subplot(gs[12:13, 5:8])
    ax56 = card.add_subplot(gs[13:14, 5:8])
    ax57 = card.add_subplot(gs[14:15, 5:8])
    ax58 = card.add_subplot(gs[15:16, 5:8])
    ax59 = card.add_subplot(gs[16:17, 5:8])
    ax60 = card.add_subplot(gs[17:18, 5:8])
    ax61 = card.add_subplot(gs[18:19, 5:8])
    ax62 = card.add_subplot(gs[19:20, 5:8])
    ax63 = card.add_subplot(gs[20:21, 5:8])
    ax64 = card.add_subplot(gs[21:22, 5:8])
    ax65 = card.add_subplot(gs[22:23, 5:8])
    ax66 = card.add_subplot(gs[23:24, 5:8])
    ax67 = card.add_subplot(gs[24:25, 5:8])
    ax68 = card.add_subplot(gs[25:26, 5:8])
    ax69 = card.add_subplot(gs[26:27, 5:8])
    ax70 = card.add_subplot(gs[27:28, 5:8])
    ax71 = card.add_subplot(gs[28:29, 5:8])
    ax72 = card.add_subplot(gs[29:30, 5:8])
    ax73 = card.add_subplot(gs[30:31, 5:8])
    ax74 = card.add_subplot(gs[31:32, 5:8])
    ax75 = card.add_subplot(gs[32:33, 5:8])
    ax76 = card.add_subplot(gs[33:34, 5:8])
    ax77 = card.add_subplot(gs[34:35, 5:8])
    ax78 = card.add_subplot(gs[35:36, 5:8])
    ax79 = card.add_subplot(gs[36:37, 5:8])
    ax80 = card.add_subplot(gs[37:38, 5:8])
    ax81 = card.add_subplot(gs[38:39, 5:8])
    ax82 = card.add_subplot(gs[39:40, 5:8])
    ax83 = card.add_subplot(gs[40:41, 5:8])
    ax84 = card.add_subplot(gs[41:42, 5:8])

    ax85 = card.add_subplot(gs[2:5, 8:11])
    ax86 = card.add_subplot(gs[5:6, 8:11])
    ax87 = card.add_subplot(gs[6:7, 8:11])
    ax88 = card.add_subplot(gs[7:8, 8:11])
    ax89 = card.add_subplot(gs[8:9, 8:11])
    ax90 = card.add_subplot(gs[9:10, 8:11])
    ax91 = card.add_subplot(gs[10:11, 8:11])
    ax92 = card.add_subplot(gs[11:12, 8:11])
    ax93 = card.add_subplot(gs[12:13, 8:11])
    ax94 = card.add_subplot(gs[13:14, 8:11])
    ax95 = card.add_subplot(gs[14:15, 8:11])
    ax96 = card.add_subplot(gs[15:16, 8:11])
    ax97 = card.add_subplot(gs[16:17, 8:11])
    ax98 = card.add_subplot(gs[17:18, 8:11])
    ax99 = card.add_subplot(gs[18:19, 8:11])
    ax100 = card.add_subplot(gs[19:20, 8:11])
    ax101 = card.add_subplot(gs[20:21, 8:11])
    ax102 = card.add_subplot(gs[21:22, 8:11])
    ax103 = card.add_subplot(gs[22:23, 8:11])
    ax104 = card.add_subplot(gs[23:24, 8:11])
    ax105 = card.add_subplot(gs[24:25, 8:11])
    ax106 = card.add_subplot(gs[25:26, 8:11])
    ax107 = card.add_subplot(gs[26:27, 8:11])
    ax108 = card.add_subplot(gs[27:28, 8:11])
    ax109 = card.add_subplot(gs[28:29, 8:11])
    ax110 = card.add_subplot(gs[29:30, 8:11])
    ax111 = card.add_subplot(gs[30:31, 8:11])
    ax112 = card.add_subplot(gs[31:32, 8:11])
    ax113 = card.add_subplot(gs[32:33, 8:11])
    ax114 = card.add_subplot(gs[33:34, 8:11])
    ax115 = card.add_subplot(gs[34:35, 8:11])
    ax116 = card.add_subplot(gs[35:36, 8:11])
    ax117 = card.add_subplot(gs[36:37, 8:11])
    ax118 = card.add_subplot(gs[37:38, 8:11])
    ax119 = card.add_subplot(gs[38:39, 8:11])
    ax120 = card.add_subplot(gs[39:40, 8:11])
    ax121 = card.add_subplot(gs[40:41, 8:11])
    ax122 = card.add_subplot(gs[41:42, 8:11])

    ax1.text(0.5,0.5, region, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontweight='bold', fontsize = 12)
    for ax, name in zip([ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29, ax30, ax31, ax32, ax33, ax34, ax35, ax36, ax37, ax38, ax39, ax40, ax41, ax42, ax43, ax44, ax45, ax46, ax47, ax85],
                    ['Variable', 'Forecast Day', 'Geopotential\nHeight', 'SLP', 'Specific\nHumidity', 'Temperature', 'U-Wind', 'V-Wind', 'Pressure\nLevel','10', '70', '100', '250', '500', '700', '850', '1000', '10', '70', '100', '250', '500', '700', '850', '10', '70', '100', '250', '500', '700', '850', '10', '70', '100', '250', '500', '700', '850', '10', '70', '100', '250', '500', '700', '850', 'COR', 'RMS']):
        if name in ['Variable', 'Pressure\nLevel', 'COR', 'RMS']:
            fw = 'bold'
        else:
            fw = 'normal'
        ax.text(0.5, 0.5, name, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight=fw, fontsize = 10)


    # Add the forecast days
    ax48.text(0, 0.5, '  1  2  3  4  5', horizontalalignment='left', verticalalignment='center', transform=ax48.transAxes, fontweight=fw, fontsize = 10)
    ax86.text(0, 0.5, '  1  2  3  4  5', horizontalalignment='left', verticalalignment='center', transform=ax86.transAxes, fontweight=fw, fontsize = 10)

    # Select all axes to remove ticks and whitespace
    ax_list = card.axes

    card.subplots_adjust(wspace=0, hspace=0)
    for ax in ax_list:
        ax.tick_params(axis='both', left=False, labelleft=False, bottom=False, labelbottom=False)

    ax1.set_facecolor((0.5, 0.5, 0.5))
    ax2.set_facecolor((0.8, 0.8, 0.8))
    ax10.set_facecolor((0.8, 0.8, 0.8))
    ax47.set_facecolor((0.8, 0.8, 0.8))
    ax85.set_facecolor((0.8, 0.8, 0.8))
    for ax_list in [[ax11, ax12, ax13, ax14, ax15, ax16, ax17], [ax19, ax20, ax21, ax22, ax23, ax24, ax25], [ax26, ax27, ax28, ax29, ax30, ax31, ax32], [ax33, ax34, ax35, ax36, ax37, ax38, ax39], [ax40, ax41, ax42, ax43, ax44, ax45, ax46]]:
        for i, ax in enumerate(ax_list):
            ax.set_facecolor((61.0/255, 97.0/255, 212.0/255, i*0.1))
    ax18.set_facecolor((61.0/255, 97.0/255, 212.0/255, i*0.1))

    #rms_ax_list = [[ax87, ax88, ax89, ax90, ax91, ax92, ax93],
    #               [ax94],
    #               [ax95, ax96, ax97, ax98, ax99, ax100, ax101],
    #               [ax102, ax103, ax104, ax105, ax106, ax107, ax108],
    #               [ax109, ax110, ax111, ax112, ax113, ax114, ax115],
    #               [ax116, ax117, ax118, ax119, ax120, ax121, ax122]]
    #cor_ax_list = [[ax49, ax50, ax51, ax52, ax53, ax54, ax55],
    #               [ax56],
    #               [ax57, ax58, ax59, ax60, ax61, ax62, ax63],
    #               [ax64, ax65, ax66, ax67, ax68, ax69, ax70],
    #               [ax71, ax72, ax73, ax74, ax75, ax76, ax77],
    #               [ax78, ax79, ax80, ax81, ax82, ax83, ax84]]

    ax_dict = {'rms': {'h': {'10': ax87, '70': ax88, '100': ax89, '250': ax90, '500': ax91, '700': ax92, '850': ax93}, 
	                   'p': {'1000': ax94}, 
                       'q': {'10': ax95, '70': ax96, '100': ax97, '250': ax98, '500': ax99, '700': ax100, '850': ax101},
                       't': {'10': ax102, '70': ax103, '100': ax104, '250': ax105, '500': ax106, '700': ax107, '850': ax108}, 
                       'u': {'10': ax109, '70': ax110, '100': ax111, '250': ax112, '500': ax113, '700': ax114, '850': ax115}, 
                       'v': {'10': ax116, '70': ax117, '100': ax118, '250': ax119, '500': ax120, '700': ax121, '850': ax122}},
               'cor': {'h': {'10': ax49, '70': ax50, '100': ax51, '250': ax52, '500': ax53, '700': ax54, '850': ax55}, 
                       'p': {'1000': ax56},
                       'q': {'10': ax57, '70': ax58, '100': ax59, '250': ax60, '500': ax61, '700': ax62, '850': ax63},
                       't': {'10': ax64, '70': ax65, '100': ax66, '250': ax67, '500': ax68, '700': ax69, '850': ax70},
                       'u': {'10': ax71, '70': ax72, '100': ax73, '250': ax74, '500': ax75, '700': ax76, '850': ax77},
                       'v': {'10': ax78, '70': ax79, '100': ax80, '250': ax81, '500': ax82, '700': ax83, '850': ax84}}}


    #for test_list, test in zip([cor_ax_list, rms_ax_list], ['cor', 'rms']):
    for test in ['cor', 'rms']:
		#for ax_list, param in zip(test_list, ['h', 'p', 'q', 't', 'u', 'v']):
        for param in ['h', 'p', 'q', 't', 'u', 'v']:
            level_key = list(data[param].keys())
            level_key.sort(key=int)
            for lev in level_key:
                dataset = data[param][lev][test]
                data_list = dataset
                for i, d in enumerate(data_list):
                    d = d[1]
                    result = get_syms(d)
                    sym = result[0]
                    color = result[1]
                    ax = ax_dict[test][param][str(lev)]
                    ax.text(0.1+0.08*i, 0.5, sym, color = color, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight=fw, fontsize = 10)

    card.savefig('{}_card_cli_temp.png'.format(region_key), dpi=500) # Save the card as png


def compile_card(exp, cont, start, end, bdate_code, edate_code, f_format):

    start = str(start)
    end = str(end)

    f_name = 'scorecard_{}_{}_{}_{}.png'.format(exp, cont, bdate_code, edate_code)

    x_t = 0.02
    l_x = 0.65

    ts = 11

    fig = plt.figure(figsize = (12,10))

    fig.suptitle('{} GEOS Scorecard'.format(exp), x = x_t, y = 0.98, fontsize=22, fontweight='bold', ha = 'left')

    plt.figtext(x_t, 0.85, '\nComparison of scores for {} (Control) and \n{} (Experiment) experiments\nfor the period of {} to {}.'.format(cont, exp, start, end), ha='left', fontsize = 15)

    plt.figtext(l_x, 0.96, 'Legend', fontweight = 'bold', fontsize = 18)
    plt.figtext(l_x, 0.92, u'\u25b2', fontsize = ts, color = 'g')
    plt.figtext(l_x+0.02, 0.92, 'far better, significant (99.99% confidence)', fontsize = ts)
    plt.figtext(l_x, 0.89, u'\u25b3', fontsize = ts, color = 'g')
    plt.figtext(l_x+0.02, 0.89, 'better, significant (99% confidence)', fontsize = ts)
    plt.figtext(l_x, 0.86, u'\u2591', fontsize = ts, color = 'g')
    plt.figtext(l_x + 0.02, 0.86, 'slightly better, significant (95% confidence)', fontsize = ts)
    plt.figtext(l_x, 0.83, u'\u2588', fontsize = ts, color = (0.7,0.7,0.7))
    plt.figtext(l_x + 0.02, 0.83, 'no significant difference', fontsize = ts)
    plt.figtext(l_x, 0.8, u'\u2591', fontsize = ts, color = 'r')
    plt.figtext(l_x + 0.02, 0.8, 'slightly worse, significant (95% confidence)', fontsize = ts)
    plt.figtext(l_x, 0.77, u'\u25bD', fontsize = ts, color = 'r')
    plt.figtext(l_x + 0.02, 0.77, 'worse, significant (99% confidence)', fontsize = ts)
    plt.figtext(l_x, 0.74, u'\u25bC', fontsize = ts, color = 'r')
    plt.figtext(l_x + 0.02, 0.74, 'far worse, significant (99.99% confidence)', fontsize = ts)


    fig.savefig(f_name, dpi=500)

    add_cards(f_name, f_format = f_format)

    return

def add_cards(f_name, f_format='PNG'):

    bg   = Image.open(f_name)
    mask = Image.new(bg.mode, bg.size, bg.getpixel((0,0)))
    diff = ImageChops.difference(bg, mask)
    bbox = diff.getbbox()
    mask.close()

    start_x = 25
    #del_x = 2000
    del_x = 500
    #y_pos = 1300
    y_pos = 325

    config_dict = {'n.hem': {'x': start_x, 'y': y_pos}, 's.hem': {'x': start_x + del_x, 'y': y_pos}, 'tropics': {'x': start_x + del_x*2, 'y': y_pos}}

    crop_l = 350
    crop_u = 350
    crop_r = 50
    crop_b = 50
    crop_diff = 50
	
    nhem = Image.open('n.hem_card_cli_temp.png')
    w, h = nhem.size
    nhem = nhem.crop((crop_l, crop_u, w-crop_r, h-crop_b))
	
    shem = Image.open('s.hem_card_cli_temp.png')
    w, h = shem.size
    shem = shem.crop((crop_l, crop_u, w-crop_r, h-crop_b))
	
    tropics = Image.open('tropics_card_cli_temp.png')
    w, h = tropics.size
    tropics = tropics.crop((crop_l, crop_u, w-crop_r, h-crop_b))
	
    nx = config_dict['n.hem']['x']
    ny = config_dict['n.hem']['y']
    sx = config_dict['s.hem']['x']
    sy = config_dict['s.hem']['y']
    tx = config_dict['tropics']['x']
    ty = config_dict['tropics']['y']
	
    # Re-size the cards on the main scorecard

    #bg = bg.resize((6000, 4800))
    bg = bg.resize((1500, 1200))

    #card_w = 2000
    #card_h = 3800
    card_w = 500
    card_h = 950

    nhem = nhem.resize((card_w, card_h))
    shem = shem.resize((card_w, card_h))
    tropics = tropics.resize((card_w, card_h))
	
    bg.paste(nhem, (nx, ny), nhem)
    bg.paste(shem, (sx, sy), shem)
    bg.paste(tropics, (tx, ty), tropics)
	
    bg.save(f_name, format=f_format)
    bg.close()

if __name__ != '__main__':
    @app.route('/v1/', methods=['GET', 'POST'])
    def api():
        # inputs (assume they exist and are correct within database)
        args_exp = flask.request.args.get('exp', None)
        args_cntrl = flask.request.args.get('cntrl', None)
    
        '''
        future extensions:
            - output = 'compact' - printer/journal quality
            - input = ['obs', 'fcst']
        '''
    
        #print(flask.request.args)
    
        # first, test db connection and find corresponding db
        try:
            db_exp, args_exp = check_db(args_exp)
        except Exception as e:
            logging.error(e)
            flask.abort(500)
        try:
            db_cntrl, args_cntrl = check_db(args_cntrl)
        except Exception as e:
            logging.error(e)
            flask.abort(500)
        print(db_exp, args_exp, db_cntrl, args_cntrl)
    
        # create db connections
        exp = scorecard.connection.Connection(db=db_exp)
        cntrl = scorecard.connection.Connection(db=db_cntrl)
    
        # get dates/times
        bdate = flask.request.args.get('bdate', None)
        edate = flask.request.args.get('edate', None)
        bdate = dt.date(int(bdate[:4]), int(bdate[4:6]), int(bdate[6:8]))
        edate = dt.date(int(edate[:4]), int(edate[4:6]), int(edate[6:8]))
    
        card = do_work(db_exp, args_exp, exp, args_cntrl, cntrl, bdate, edate)
    
        # with an api, we should return only a single score block as a json object and let other routes use the api in
        # conjunction with forms and user interactivity
        verify = 'self'
        if '.21z' in args_exp:
            args_exp = args_exp[:-4]
        if '.21z' in args_cntrl:
            args_cntrl = args_cntrl[:-4]
        if '_ec' in args_exp:
            args_exp = args_exp[:-3]
            verify = 'ecwmf'
        if '_ec' in args_cntrl:
            args_cntrl = args_cntrl[:-3]
            verify = 'ecmwf'
        # [args_exp if 'x0035' not in args_exp else 'x0037_noSPPT'][0]
        bdate = bdate.strftime('%B %d, %Y').replace(' 0', ' ')
        edate = edate.strftime('%B %d, %Y').replace(' 0', ' ')
        return flask.render_template(
                'scorecard/landing.html', data=json.loads(json.dumps(card, indent=4, sort_keys=True, separators=(',', ': '))),
                service="GEOS Scorecard", exp=args_exp, bdate=bdate, edate=edate, cntrl=args_cntrl, verify=verify,
            )

def do_work(db_exp, args_exp, exp, args_cntrl, cntrl, bdate, edate):
    # scorecard parameters
    fields = ['t', 'h', 'u', 'v', 'q', 'p']
    domains = ['n.hem', 's.hem', 'tropics']
    stats = ['cor', 'rms']
    # pressure levels are different per experiment (and version of DAS)
    if not check_db_lev(exp, args_exp, fields, domains, stats) or not check_db_lev(cntrl, args_cntrl, fields, domains, stats):
        levels = [850, 700, 500, 250, 100]
    else:
        levels = [850, 700, 500, 250, 100, 70, 10]

    #fields = ['h']
    #domains = ['s.hem']
    #stats = ['cor']
    #levels = [500]

    '''
    future
    Geopotential Heights @ (100, 250, 500, 850) HPa
    SLP
    Q @ (10, 70, 100, 250, 500, 850) HPa
    Temperature @ (10, 70, 100, 250, 500, 850) HPa
    U/V @ (10, 70, 100, 250, 500, 850) HPa
    Q @ (10, 70, 100, 250, 500, 850) HPa
    T2M
    U10m and V10m
    Ozone @ (10, 70) HPa
    '''

    # dictionary to contain all of scorecard results
    card = {}

    # iterate over confidence levels
    for l,lev in enumerate([.9999, .99, .95]): # changed from Monitoring Meeting request on 2/15/2019
        for field, level, domain, stat in list(itertools.product(*[fields, levels, domains, stats])):
            # only surface pressure
            if field in 'p':
                level = 1000
            # fill in dictionary with empty lists
            if domain in card:
                if field in card[domain]:
                    if level in card[domain][field]:
                        if stat not in card[domain][field][level]:
                            card[domain][field][level][stat] = card[domain][field][level].get(stat, [])
                    else:
                        card[domain][field][level] = card[domain][field].get(level, {stat: []})
                else:
                    card[domain][field] = card[domain].get(field, {level: {stat: []}})
            else:
                card[domain] = card.get(domain, {field: {level: {stat: []}}})

    # set forecast length
    fcst_length = list(range(24, 5*24+1, 12))
    if 'ops' in db_exp and 'ec' not in args_exp and 'ec' not in args_cntrl and 'gfs' not in [args_exp, args_cntrl] and 'fpp' not in args_exp and 'fpp' not in args_cntrl and 'rp' not in args_exp and 'rp' not in args_cntrl and 'fp' not in args_exp and 'fp' not in args_cntrl:
        fcst_length = range(24, 8*24+1, 24)

    # get dates strings
    dates = []
    for i in range((edate - bdate).days+1):
        dates.append(int((bdate + dt.timedelta(days=1*i)).strftime('%Y%m%d00')))

    # obtain all the data
    for l,lev in enumerate([.9999, .99, .95]): # changed from Monitoring Meeting request on 2/15/2019
        for field, level, domain, stat in list(itertools.product(*[fields, levels, domains, stats])):
            if field in 'p':
                level = 1000

            # retrieve per date
            e_data = []
            c_data = []
            for date in dates:
                e = exp.get(
                  args_exp,
                  **{
                      'variable':     field,
                      'level':        level,
                      'domain_name': domain,
                      'statistic':     stat,
                      'step':   fcst_length,
                      'date':          date,
                  }
                )
    
                c = cntrl.get(
                  args_cntrl,
                  **{
                      'variable':     field,
                      'level':        level,
                      'domain_name': domain,
                      'statistic':     stat,
                      'step':   fcst_length,
                      'date':          date,
                      #'verify': 'ecmwf',
                  }
                )
    
                # if step == 0:
                #     continue
                # need to accept delta t
                # if no dates are given, use what is in the db that is common between exp and cntrl
    
                # there could be a mismatch to the exp/cntrl dates
                dates_e = [x for x,y in e]
                dates_c = [x for x,y in c]
                diff = set(dates_e).symmetric_difference(set(dates_c))
                if diff:
                    print('missing date(s) in database:' + str(sorted(diff)))
                    print('experiment: ', str(args_cntrl if list(diff)[0] not in dates_c else args_exp))
                    print(field, level, domain, stat) #, step)
                    try:
                        flask.abort(500)
                    except:
                        sys.exit(1)
    
                #dates = [x for x,y in e for a,z in c if x==a]
                values_e = np.array([[y] for x,y in e])
                values_c = np.array([[z] for a,z in c])
                # e = sorted(e)
    
                if stat in 'rms':
                    score = RootMeanSquare()
                else:
                    score = Correlation()
    
                # test for empty sets
                if not len(values_e):
                    print('experiment',field, level, domain, stat, date)
                if not len(values_c):
                    print('control',field, level, domain, stat, date)

                e_data.append(np.stack(values_e, 1)[0])
                c_data.append(np.stack(values_c, 1)[0])

            # need to stack these due to Will's code only doing a 
            #for i in fcst_length:
            sig = score.significance(
                np.ma.masked_values(c_data, 1.7e+38),
                np.ma.masked_values(e_data, 1.7e+38),
                lev=lev,
            )

            s, lower, upper = sig
            #if stat in 'cor' and domain in 's.hem' and field in 'h' and level == 500:
            #    print(sig)
            if stat in 'rms':
                s = score.mean(s) # calculates the mean for each step (old GMAOPy anom cor code)

            #print(lev, s, lower)

            significant = False
            for i,step in enumerate(fcst_length):
                # is the significance larger than either significance box?
                if np.abs(s[i]) >= np.abs(upper[i]):
                    significant = True

                    # determine if significance is good or bad
                    s_sign = np.sign(s[i])
                    u_sign = np.sign(upper[i])

                    # assume that significance boxes are centered at zero
                    # cor: worse if more negative than box

                    if s_sign < 0:
                        # cor worse (what about rms?)
                        # card slot for step empty
                        if step not in [a for a,b in card[domain][field][level][stat]]:
                            card[domain][field][level][stat].append((step, -3+l))
                        else:
                            this = [(a,b) for a,b in card[domain][field][level][stat] if a == step][0]
                            if not this[1]:
                                # value still zero: check to see if significant at current level
                                card[domain][field][level][stat].remove(this)
                                card[domain][field][level][stat].append((step, -3+l))
                    else:
                        if step not in [a for a,b in card[domain][field][level][stat]]:
                            card[domain][field][level][stat].append((step, 3-l))
                        else:
                            this = [(a,b) for a,b in card[domain][field][level][stat] if a == step][0]
                            if not this[1]:
                                # value still zero: check to see if significant at current level
                                card[domain][field][level][stat].remove(this)
                                card[domain][field][level][stat].append((step, 3-l))
                else:
                    if step not in [a for a,b in card[domain][field][level][stat]]:
                        card[domain][field][level][stat].append((step, 0))
                    else:
                        this = [(a,b) for a,b in card[domain][field][level][stat] if a == step][0]
                        if not this[1]:
                            card[domain][field][level][stat].remove(this)
                            card[domain][field][level][stat].append((step, 0))
            # instead of appending, let's send it to a method to determine its score instead and insert into the card

    for domain in card:
        for field in card[domain]:
            for level in card[domain][field]:
                for stat in card[domain][field][level]:
                    card[domain][field][level][stat] = sorted(card[domain][field][level][stat], key=lambda tup: tup[0])

    return card






if __name__ != '__main__':
    @app.route('/<api>/', methods=['GET', 'POST'])
    def api2(api='blah'):
        return flask.render_template(
                'construction.html',
                service="GEOS Scorecard",
            )

if __name__ == '__main__':
    # parse args and run api method
    sys.exit(main(sys.argv[1:]))
