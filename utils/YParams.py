from ruamel.yaml import YAML
import logging

class YParams():
  """ Yaml file parser """
  def __init__(self, yaml_filename, config_name, print_params=False):
    self._yaml_filename = yaml_filename
    self._config_name = config_name
    self.params = {}

    if print_params: 
      print("------------------ Configuration ------------------")

    with open(yaml_filename) as _file:

      for key, val in YAML().load(_file)[config_name].items():
        if print_params: print(key, val)
        if val =='None': val = None

        self.params[key] = val
        self.__setattr__(key, val)

    if print_params:
      print("---------------------------------------------------")

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val

  def log(self):
    logging.info("------------------ Configuration ------------------")
    logging.info("Configuration file: "+str(self._yaml_filename))
    logging.info("Configuration name: "+str(self._config_name))
    with open(self._yaml_filename) as yamlfile:
      for key, val in YAML().load(yamlfile)[self._config_name].items():
        logging.info(str(key) + ' ' + str(val))
    logging.info("---------------------------------------------------")
