from progress.bar import IncrementalBar

UPDATE_CONST = 4
bar = IncrementalBar('File loaded: ', max=1000, suffix='%(percent)d%%')

def restartBar(max_steps=0):
    bar = IncrementalBar('File loaded: ', max=max_steps, suffix='%(percent)d%%')

def next():
    bar.next()

def resetBar():
    bar.goto(0)