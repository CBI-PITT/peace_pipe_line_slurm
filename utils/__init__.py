import os


def get_user(path):
    user = None
    try:
        user = path.replace('/bil/users/', '').split('/')[0]
    except:
        pass
    if user:
        return user
    return "lab"
