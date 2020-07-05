import os
import glob
from datetime import datetime
from pickle import dump, load

from billys.util import get_data_home


def save(step: int, obj, data_home=None):
    """
    Save the given object as checkpoint. 

    Parameters
    ----------
    step: int, required
        The checkpoint step.

    data_home : optional, default: None
        A path for checkpoints object. 
        If None, `DEFAULT_PATH/checkpoints` will be used.
    """
    now = datetime.now()
    now_str = now.strftime('%Y%m%d-%H%M%S-%f')

    if data_home is None:
        path = os.path.join(get_data_home(), 'checkpoints')
    else:
        path = data_home
    name = f'{now_str}_step-{step}.pkl'

    dump(obj, open(f'{os.path.join(path, name)}', 'wb'))

    return name


def revert(step: int, data_home=None):
    """
    Load the last dataset checkpoint for the step `step`.

    Parameters
    ----------
    step: int, required
        The checkpoint step.

    data_home : optional, default: None
        A path for checkpoints object. 
        If None, `DEFAULT_PATH/checkpoints` will be used.
    """

    if data_home is None:
        path = os.path.join(get_data_home(), 'checkpoints')
    else:
        path = data_home

    # search for pkl files in dir
    checkpoints = glob.glob(f'{path}/*.pkl')
    # remove the extension
    checkpoints = [checkpoint.split('.')[0]
                   for checkpoint in checkpoints]
    # split date and step
    checkpoints = [checkpoint.split('_') for checkpoint in checkpoints]
    # keep only filenames with desidered step
    checkpoints = [checkpoint[0]
                   for checkpoint in checkpoints if checkpoint[1] == f'step-{step}']
    # sort in non increasing order
    checkpoints.sort(reverse=True)
    # keep the first result, i.e., the newest checkpoint
    last = checkpoints[0]

    # assemble the checkpoint filename
    name = f'{last}_step-{step}.pkl'

    return load(open(f'{os.path.join(path, name)}', 'rb'))
