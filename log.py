from datetime import datetime


LOG_FILE = './logs.txt'

def log_progress(message):
    now = datetime.now()
    timestamp_format = '%Y-%h-%d-%H:%M:%S'
    timestamp = now.strftime(timestamp_format)
    with open(LOG_FILE,'a') as f:
        f.write(timestamp + ' : ' + message + '\n')