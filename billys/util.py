import os
import os.path

BILLYS_WORKSPACE_NAME = '.billys'


def get_data_home(data_home=None):
    """
    Returns
    ------- 
    path: str
        `data_home` if it is not None, otherwise a path to a directory into the 
        user HOME path.
    """

    if data_home is None:

        user_home = os.path.expanduser('~')
        if user_home is None:
            raise RuntimeError(
                'You should specify at least your home directory with HOME env variable.')

        return f'{os.path.join(user_home, BILLYS_WORKSPACE_NAME)}'

    return data_home
