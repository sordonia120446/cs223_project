import subprocess

try:
    import hmmlearn
except ImportError:
    print('Installing hmmlearn')
    subprocess.call(['pip3', 'install', 'hmmlearn'])


print('')
print('hmmlearn version: {}'.format(hmmlearn.__version__))
print('')
print('These are all the installed python packages.')
subprocess.call(['pip3', 'list'])
