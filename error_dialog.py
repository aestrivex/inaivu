from traitsui.message import error

class InaivuError(ValueError): 
    pass

def error_dialog(message):
    error(message)
    raise InaivuError(message)
