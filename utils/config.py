import os
from sys import exit

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import yaml
from easydict import EasyDict

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file
    :param yaml_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config yaml file provided
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.safe_load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            logging.getLogger().warning(
                "INVALID YAML file format! Please provide a good yaml file.")
            exit()


def process_config(yaml_file):
    """
    Get the yaml file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param yaml_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_yaml(yaml_file)

    # making sure that the experiment name is provided.
    try:
        logging.getLogger().info("Experiment: {}.".format(config.experiment.name))
    except AttributeError:
        logging.getLogger().warning("ERROR! Please provide the experiment name in yaml file.")
        exit()
    
    # create some important directories to be used for the experiment.
    logging.getLogger().info("Loading configuration and creating dirs.")
    config.dirs.summary_dir = os.path.join(config.dirs.base_dir, config.experiment.name, "summaries/")
    config.dirs.checkpoint_dir = os.path.join(
        config.dirs.base_dir, config.experiment.name, "checkpoints/")
    config.dirs.out_dir = os.path.join(
        config.dirs.base_dir, config.experiment.name, "out/")
    config.dirs.log_dir = os.path.join(
        config.dirs.base_dir, config.experiment.name, "logs/")
    create_dirs([config.dirs.summary_dir, config.dirs.checkpoint_dir, config.dirs.out_dir, config.dirs.log_dir])

    # setup logging in the project
    setup_logging(config.dirs.log_dir)
    
    if(config.settings.verbose):
        logging.getLogger().info(" The Configuration of the experiment: ")
        logging.getLogger().info(config)

    return config

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info(
            "Creating directories error: {0}".format(err))
        exit()
