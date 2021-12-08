from progress.bar import IncrementalBar
from settings.settings_access import settings 

UPDATE_CONST = 4
bar = IncrementalBar('Elapsed time', max=settings.sim_time*UPDATE_CONST, suffix='%(percent)d%%')

def resetBar():
    bar.goto(0)