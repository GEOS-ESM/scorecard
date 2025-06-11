from __future__ import print_function

import psycopg2
#import psycopg2.sql as sql
from . import config
import flask

# need to improve error handling and propagation of rc

class Connection(object):
    '''a basic PostGreSQL connection'''

    def __init__(self, host=config.host, db=None):
        # only 2 db available presently
        if db not in config.db:
            print('Database name provided not in list of known profiles.')
            print(db)
            flask.abort(500)
            #sys.exit(1)
        self.db = db
        try:
            # note: no port specified
            # note: no connect_timeout specified (i.e., waits indefinitely)
            # note: assumes .pgpass is available

            # version can differ between (dbname/database)
            print(host, config.db[self.db], config.user)
            self.con = psycopg2.connect(
                host=host, database=config.db[self.db],
                user=config.user
            )
        except psycopg2.OperationalError as e:
            try:
                # try default user if run from CLI
                self.con = psycopg2.connect(
                    host=host, database=config.db[self.db],
                    user='gmao_user'
                )
                return
            except psycopg2.OperationalError as e:
                print(self.db)
                print(config.db[self.db])
                print('Failed to connect to database.\n{0}'.format(e))
                sys.exit(2)
            print(self.db)
            print(config.db[self.db])
            print('Failed to connect to database.\n{0}'.format(e))
            flask.abort(500)

    def check(self, expver=None):
        '''Check to verify expver is found in database by retrieving the first
        row from self.db connection that satisfies the exists query.'''
        if expver is None:
            return []

        # can I optimize this?
        # https://danmartensen.svbtle.com/sql-performance-of-join-and-where-exists
        # https://thoughtbot.com/blog/advanced-postgres-performance-tips
        # https://stackoverflow.com/questions/7471625/fastest-check-if-row-exists-in-postgresql
        exists_query = '''
            select exists (
                select *
                from {0}.v_view
                where expver = %s limit 1
            )
        '''
        cursor = self.con.cursor()
        # need an alternative to .format
        cursor.execute(
            exists_query.format(self.db), (expver,)
        )
        row = cursor.fetchone()
        if row:
            row = row[0]

        return row

    def get(self, expver=None, num=None, **kwargs):
        '''Generic database querying of forecast skill scores.'''
        if expver is None:
            return []

        # build query
        query = 'select date, value from ' + self.db + '.v_view where expver=%s '
        args = [expver]
        for key,value in kwargs.items():
            query += ' and '
            if isinstance(value, list):
                query += key + ' in %s'
                args += [tuple([v for v in value])]
            else:
                query += key + '=%s'
                args += [value]

        cursor = self.con.cursor()
        if num:
            cursor.execute(query + ' limit ' + str(int(num)) + ';', tuple(args))
        else:
            cursor.execute(query + ';', tuple(args))
        rows = cursor.fetchall()

        cursor.close()
        return rows
