# GMAO scorecard

## Usage

(Assuming this is running on NCCS Discover SLES15, from the root directory of this repository.)

First, load modules:


```sh
source $PWD/modules_3
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
