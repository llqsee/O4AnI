import sys
mods = {
    'numpy': 'np',
    'pandas': 'pd',
    'matplotlib': 'mpl',
    'seaborn': 'sns',
    'scipy': 'scipy',
    'sklearn': 'sklearn',
    'statsmodels': 'statsmodels',
}

print('PYTHON:', sys.executable)
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'{m}: OK {ver}')
    except Exception as e:
        print(f'{m}: FAIL {e!r}')