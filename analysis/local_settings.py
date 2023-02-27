import json


def update_settings(settings, config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
        settings.PYTORCH_MODEL_PATH = config.get("PYTORCH_MODEL_PATH")
        settings.PYTORCH_MODEL_NAME = config.get("PYTORCH_MODEL_NAME")
        settings.DB_LOCATION = config.get("DB_LOCATION")
        settings.DB_TYPE = config.get("DB_TYPE")
        settings.MYSQL_DB_NAME = config.get("MYSQL_DB_NAME")
        settings.PYTORCH_MODEL_VERSION = config.get("PYTORCH_MODEL_VERSION")
        settings.DATAFRAME_VERSION = config.get("DATAFRAME_VERSION")

    # print("Settings file", config_file)
    # with open(config_file, "r") as f:
    #     local_settings_str = f.read()
    #     local_settings = json.loads(local_settings_str)
    #     print("local_settings", local_settings)
    # for k, v in local_settings.items():
    #     setattr(settings, k, v)