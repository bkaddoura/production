# this pattern is common to all Jesus's code
import logging
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()

# injected from environments file, create a subdirectory and add there
LOG_DIR = os.getenv('LOG_DIR', './logs/')

# will allow you to record more or less details; debug are messages I need to debug code
# use a debug statement and keep it in your code
# warnings highlighted in orange
# errors in red, we can raise errors in our logs by calling ERROR
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def get_logger(name, log_dir = LOG_DIR, log_level = LOG_LEVEL):

    '''
    Set up a logger with the given name and log level.
    '''
    _logs = logging.getLogger(name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # this is how it's structured
        # tell me the Name, Filename, Line Number, and messages passed
        # different strategies, ususally timestamp it
        # can setup in rotating sequence
        # not meant to be kept for ever, to help in production and stored temperarly
        # meant to to be kept in files
    f_handler = logging.FileHandler(os.path.join(log_dir, f'{ datetime.now().strftime("%Y%m%d_%H%M%S") }.log'))
    f_format = logging.Formatter('%(asctime)s, %(name)s, %(filename)s, %(lineno)d, %(funcName)s, %(levelname)s, %(message)s')
    f_handler.setFormatter(f_format)
    _logs.addHandler(f_handler)
    
    # stream is your standard stream
    s_handler = logging.StreamHandler()
    s_format = logging.Formatter('%(asctime)s, %(filename)s, %(lineno)d, %(levelname)s, %(message)s')
    s_handler.setFormatter(s_format)
    _logs.addHandler(s_handler)
    
    _logs.setLevel(log_level)
    return _logs