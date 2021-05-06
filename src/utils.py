import logging
import os
import json
import glob

class Dictobj(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getitem__(self, key):
        return getattr(self, key)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Dictobj(value) if isinstance(value, dict) else value

def load_config(conf):
    with open(os.path.join("config", "{}.json".format(conf)), "r") as f:
        config = json.load(f)
    return Dictobj(config)