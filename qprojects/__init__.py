# -*- coding: utf-8 -*
import os as _os
import logging as _logging
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove tf compulation warnings
_tf_logger = _logging.getLogger('tensorflow')
_tf_logger.setLevel(_logging.DEBUG)  # remove tf logs
