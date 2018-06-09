# -*- coding: utf-8 -*
import os as _os
import logging as _logging
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # remove tf compulation warnings
_tf_logger = _logging.getLogger('tensorflow')
_tf_logger.setLevel(_logging.DEBUG)  # remove tf logs

# pylint: disable=wrong-import-position
from ._players import IntelligentPlayer
from ._players import BasicNetwork
from ._game import DefaultPlayer
from ._game import initialize_players_cards
from ._game import play_game
from ._game import GameBoard
