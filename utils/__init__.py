import os


def get_user(path):
    try:
        user = path.replace('/h20/Public/', '').split('/')[0]
    except:
        user = "lab"
    return user
