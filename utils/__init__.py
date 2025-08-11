import os


def get_user(path):
    user = None
    try:
        user = path.replace('/h20/Public/', '').split('/')[0]
    except:
        pass
    if user:
        return user
    return "lab"
