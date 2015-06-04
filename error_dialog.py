from traitsui.message import error

def InaivuError(ValueError): pass

def error_dialog(message):
    error(message)
    raise InaivuError(message)
