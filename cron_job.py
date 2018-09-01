import os
import time
import sys


def delete_daily():
    '''
    This method is called once a day. It deletes any file in the the 'uploads', 'uploads/extracted' and 'pickles' directories that is older than 24 hours.
    This makes sure that no user files remain on the system.

    '''

    print()
    print('>>>>>>>>>>>>>> CRON JOB CALLED <<<<<<<<<<<<', file=sys.stdout)
    print()

    for filename in os.listdir('uploads'):
        path = os.path.join('uploads', filename)

        age = time.time() - (os.path.getmtime(path))
        if filename != 'extracted' and age >= 86400.0:
            os.remove(path)

    for filename in os.listdir('pickles'):
        path = os.path.join('pickles', filename)

        age = time.time() - (os.path.getmtime(path))
        if age >= 86400.0:
            os.remove(path)

    for filename in os.listdir('uploads/extracted'):
        path = os.path.join('uploads/extracted', filename)

        age = time.time() - (os.path.getmtime(path))
        if age >= 86400.0:
            os.remove(path)
