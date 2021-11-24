from progress.bar import IncrementalBar

UPDATE_CONST = 4
bar = IncrementalBar('File loaded: ', max=1000, suffix='%(percent)d%%')

def resetBar():
    bar.goto(0)