import scorecard
import logging

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

    # can I check both cases at once to optimize for performance?
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
