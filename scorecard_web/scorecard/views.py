

import logging
logging.basicConfig(level='WARN') # put in config?

import os
import flask
import datetime as dt
import json
import numpy as np
import itertools
from scipy.special import stdtr

import sys
import psycopg2

#from . import app

from . import scorecard
from . import v2

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

#@app.route('/v1/', methods=['GET', 'POST'])
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
    # OPS = 8 days
    # EXP = 5 days
    fcst_length = list(range(24, 5*24+1, 12))
    #fcst_length = [4*24-12, 4*24, 4*24+12, 5*24]
    if 'ops' in db_exp and 'ec' not in args_exp and 'ec' not in args_cntrl and 'gfs' not in [args_exp, args_cntrl] and 'fpp' not in args_exp and 'fpp' not in args_cntrl and 'rp' not in args_exp and 'rp' not in args_cntrl and 'fp' not in args_exp and 'fp' not in args_cntrl:
        fcst_length = list(range(24, 8*24+1, 24))
    #if 'ops' in db_exp and [i for i in [args_exp, args_cntrl] if 'ec' in i or 'fpp' in i or 'fpp' in i] and 'gfs' not in [args_exp, args_cntrl]:

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
                    flask.abort(500)
    
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

#@app.route('/db_check_tool/', methods=['GET', 'POST'])
def db_check_tool():
    exp_dates = None
    exp_options = None
    try:
        exp_dates = str(flask.request.form['exp_dates'])
        if exp_dates:
            date_range = db_date_range(exp_dates)
            if date_range:
                pass
            else:
                date_range = ['No data available for this experiment']
        else:
            date_range = None
    except:
        date_range = None

    # Get experiment names from wildcard

    try:
        exp_options = str(flask.request.form['exp_options'])
        if exp_options:
            ev_list = db_expver_list(exp_options)
            if ev_list:
                pass
            else:
                ev_list = ['No experiments available']
        else:
            ev_list = None
    except:
        ev_list = None

    # URL Generation

    try:
        exp_name = str(flask.request.form['exp'])
        ctrl_name = str(flask.request.form['ctrl'])
        bdate_val = str(flask.request.form['bdate'])
        edate_val = str(flask.request.form['edate'])
    except:
        exp_name = None
        ctrl_name = None
        bdate_val = None
        edate_val = None

    return flask.render_template('scorecard/db_check.html', exp_dates = exp_dates, date_range = date_range, exp_options = exp_options, ev_list = ev_list, exp_name = exp_name, ctrl_name = ctrl_name, bdate_val = bdate_val, edate_val = edate_val)

def db_date_range(exp):
    conn = scorecard.connection.Connection(db='fc_exp', host='edb1')
    cur = conn.con.cursor()

    query = "SELECT distinct date from fc_exp.v_view where expver = '{}' ORDER BY date".format(exp)
    cur.execute(query)
    val = cur.fetchall()

    if val:
        pass
    else:
        date_list = []
        return date_list

    start = dt.datetime.strptime('{}'.format(np.min(val)), "%Y%m%d%H")
    end = dt.datetime.strptime('{}'.format(np.max(val)), "%Y%m%d%H")

    ldate = None
    sdate = start
    date_list = []
    for date in val:
        cdate = dt.datetime.strptime('{}'.format(np.min(date)), "%Y%m%d%H")
        if ldate is not None:
            delta = (cdate - ldate).days
            if (delta > 1):
                print(sdate.strftime("%Y%m%d"),'-',ldate.strftime("%Y%m%d"))
                date_range = str(sdate.strftime("%Y%m%d") + '-' + ldate.strftime("%Y%m%d"))
                date_list.append(date_range)
                ldate = None
                sdate = cdate
            else:
                ldate = cdate
        else:
            ldate = cdate


    date_range = str(sdate.strftime("%Y%m%d") + '-' + ldate.strftime("%Y%m%d"))

    if len(date_list) == 0:
        date_list.append(date_range)
    elif date_list[-1] != date_range:
        date_list.append(date_range)
    else:
        pass

    return date_list

def db_expver_list(exp):
    ev_list = []

    conn = scorecard.connection.Connection(db='fc_exp', host='edb1')
    cur = conn.con.cursor()

    query = "SELECT distinct expver from fc_exp.v_view where expver LIKE '%{}%'".format(exp)
    cur.execute(query)
    val = cur.fetchall()

    ev_list = []

    for x in np.sort(val):
        ev_list.append(x[0])

    return ev_list

#@app.route('/<api>/', methods=['GET', 'POST'])
#def api2(api='blah'):
#    return flask.render_template(
#            'construction.html',
#            service="GEOS Scorecard",
#        )
