# GMAO scorecard

The GMAO Scorecard tool allows users to evaluate model experiments.

## Prerequisites

Scorecard usage currently requires read (`select`) access to the `gmao_stats` (for `fc_ops`) and `semper` (`fc_exp`) PostgreSQL databases on `edb1`.
The username and password for those must be configured in your `~/.pgpass` file for scorecard to work.

## Command-line usage

(Assuming this is running on NCCS Discover SLES15, from the root directory of this repository.)

First, load modules:


```sh
source $PWD/modules_3
```

Next, move into the `./scorecard_web/scorecard` subdirectory:

```sh
cd scorecard_web/scorecard
```

Some example invocations:

```sh
# Gencast
python ./views.py --exp GenCast --cntrl f5295_fp --bdate 2024050100 --edate 2024053100
python ./views.py --exp GenCast --cntrl f5295_fp --bdate 2024120100 --edate 2024123100
# Aifs
python ./views.py --exp Aifs --cntrl f5295_fp --bdate 2024050100 --edate 2024053100
python ./views.py --exp Aifs --cntrl f5295_fp --bdate 2024120100 --edate 2024123100

# Aurora
python ./views.py --exp Aurora --cntrl f5295_fp --bdate 2024050100 --edate 2024053100
python ./views.py --exp Aurora --cntrl f5295_fp --bdate 2024120100 --edate 2024123100

# Pangu
python ./views.py --exp Pangu --cntrl f5295_fp --bdate 2024050100 --edate 2024053100
python ./views.py --exp Pangu --cntrl f5295_fp --bdate 2024120100 --edate 2024123100

# PrithviAI
python ./views.py --exp PrithviAI --cntrl f5295_fp --bdate 2024050100 --edate 2024053100
python ./views.py --exp PrithviAI --cntrl f5295_fp --bdate 2024120100 --edate 2024123100
```

## Alternate installation: Python and uv

Make sure `uv` is installed (user-level install to home directory; no admin required): https://docs.astral.sh/uv/getting-started/installation/.

Create a virtual environment:

```sh
uv venv
```

Install requirements:

```sh
uv pip install -r pyproject.toml
```

Activate the environment:

```sh
source .venv/bin/activate
```

Now, you should be able to run the `python` commands above.

## About the database

For a given schema (e.g., `fc_ops`), there's a view definition, `v_view`, defined as something like:

```sql
 SELECT domain.north,
    domain.west,
    domain.south,
    domain.east,
    domain.domain_name,
    stats.level,
    stats.levtype,
    stats.expver,
    stats.source,
    stats.forecast,
    stats.verify,
    stats.statistic,
    stats.type,
    stats.variable,
    value.date,
    value.step,
    value.value,
    value.count
   FROM fc_ops.domain,
    fc_ops.stats,
    fc_ops.value
  WHERE stats.domain_id = domain.id AND value.stats_id = stats.id;
```

There are ~2.3 billion rows in `value` and 300K rows in `stats`.

## Implementation notes

Most of the work of the scorecard is done in the `views.py` file, in the `do_work` function.

### Pick the right database

The scorecard will look for the experiment in two databases --- `gmao_stats` (for table `fc_ops`) and `semper` (for table `fc_exp`).
In both cases, the data is stored in a virtual table ("view") called `v_view`.

### Retrieving the data

The relevant code looks like this.

```python
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
```

`exp` is a `Connection` object defined in `scorecard/connection.py`.
The `get` method turns this dict into an SQL query through string concatenation.
The query above is effectively:

```sql
SELECT date, value FROM fc_ops.v_view
WHERE expver = <experiment>
AND variable = <field>
AND level = <level>
AND domain_name = <domain_name>
# etc..
LIMIT <num>   # optional
```

`e` is set to the results for the experiment; `c` is set to the results for the control.

### Calculate scores

Use the `Correlation` or `RootMeanSquare` objects to calculate the corresponding statistic relating the experiment and the control.
