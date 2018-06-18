# -*- coding: utf-8 -*
from pathlib import Path
import ujson as json
import numpy as np
from . import _deck
from . import _utils


_BONUS_CARDS = {_deck.Card(c) for c in ["Qh", "Kh"]}


class DefaultPlayer:  # pylint: disable=too-many-instance-attributes
    """Player which selects one card randomly at each round.

    Note
    ----
    This class is the base class for other players. They should mostly improve
    methods "set_reward" and "_propose_card_to_play". A proposition can be rejected if
    it proposed an card which could not be played. A random choice is made in this case,
    and the acceptation_ratio attribute allows to keep track of how often the propositions
    are accepted.
    """

    def __init__(self):
        self._cards = _deck.CardList([])
        self._initial_cards = _deck.CardList([])  # keep a record of initial cards for each game
        self._order = None
        self._card_played_count = 0
        self._erroneous_selection_count = 0
        self._last_playable_cards = None
        self._acceptation_queue = _utils.ReplayQueue(1000)
        self.reward_sum = 0

    def reinitialize(self):
        self._cards = _deck.CardList([])
        self._last_playable_cards = None
        self._initial_cards = _deck.CardList([])  # keep a record of initial cards for each game
        self._order = None

    def get_acceptation_ratio(self):
        """Ratio of card proposition which have been accepted (allowed to play)
        """
        return (self._card_played_count - self._erroneous_selection_count) / self._card_played_count

    def get_instantaneous_acceptation_ratio(self):
        """Ratio of card proposition which have been accepted (allowed to play)
        """
        return np.mean(self._acceptation_queue._data)

    @property
    def initial_cards(self):
        return self._initial_cards

    @property
    def order(self):
        return self._order

    @property
    def cards(self):
        """Makes a copy, to make sure no modification happens outside
        """
        return _deck.CardList(self._cards, self._cards.trump_suit)

    def initialize_game(self, order, cards):
        """Initialize a game with order and cards.

        Parameters
        ----------
        order: int
            the order in which the player will play
        cards: list
            a list of 8 cards

        Note
        ----
        A game can only be initialized if the card list is empty (no ongoing game)
        """
        assert not self._cards, "Cannot initialize a new game when card are still at play: {}".format(self._cards)
        assert len(cards) == 8, "Wrong number of cards for initialization: {}.".format(self._cards)
        self._cards = cards
        self._initial_cards = _deck.CardList(cards)
        self._order = order

    def _get_playable_cards(self, board):
        """Returns the cards that can be played
        """
        if self._cards.trump_suit is None:
            self._cards.trump_suit = board.trump_suit
        round_cards = board.get_current_round_cards()
        return self._cards.get_playable_cards([] if len(round_cards) == 4 else round_cards)

    def get_card_to_play(self, board):
        """Returns an acceptable card to play.

        Parameter
        ---------
        board: GameBoard
            the current board for the game

        Returns
        -------
        Card
            an acceptable card to play in the current game

        Note
        ----
        This function makes sure the sent card is acceptable to play. It keeps tracks of remaining
        cards, and of how often the propositions (from a neural network for instance) where accepted.
        Propositions are provided through the "_propose_card_to_play" method.
        The playable cards at this round are kept for later use in set_reward.
        """
        selected = self._propose_card_to_play(board)
        self._last_playable_cards = self._get_playable_cards(board)
        if selected is None or selected not in self._last_playable_cards:
            #print(np.round(self._get_expectations(board)), len(board.actions))
            self._erroneous_selection_count += 1
            self._acceptation_queue.append(False)
            selected = np.random.choice(self._last_playable_cards)
        else:
            self._acceptation_queue.append(True)
        self._cards.remove(selected)
        self._card_played_count += 1
        return selected

    def set_reward(self, board, value):  # pylint: disable=unused-argument
        """Function to be called after each action on the board, to provide feedbacks for neural networks
        for instance.

        Parameter
        ---------
        board: GameBoard
            the current board for the game
        value: int
            the value of the reward

        Note
        ----
        This function is called after *each* action (from any player), while get_card_to_play method
        is only called when it is the user's time to play.
        """
        self.reward_sum += value

    def _propose_card_to_play(self, board):  # pylint: disable=unused-argument
        """Propose a card to play thanks to an advanced method.

        Parameter
        ---------
        board: GameBoard
            the current board for the game

        Returns
        -------
        Card
            a card proposition for playig, which may be unacceptable.

        Note
        ----
        Implement a technique here.
        """
        pass


def initialize_players_cards(players):
    """Initialize players for a new game.
    This function sets the player order and its cards.

    Parameter
    ---------
    player: list
        a list of 4 players.
    """
    assert len(players) == 4
    # initialize players' cards
    cards = _deck.get_full_deck()
    np.random.shuffle(cards)
    for k, cards in enumerate(_utils.grouper(cards, 8)):
        players[k].initialize_game(k, _deck.CardList(cards))


def play_game(board, players, verbose=False):
    """Plays a game, given a board with biddings and initialized players.

    Parameters
    ----------
    board: GameBoard
        a board, with biddings already performed, but no action
    players: list
        a list of 4 initialized players, with 8 cards each and given orders
    """  # IMPROVEMENT: handle partially played games
    # checks
    assert board.biddings, "Biddings must have been already performed"
    assert not board.actions, "No cards should have already been played"
    for k, player in enumerate(players):  # make sure the data is correct
        assert player._order == min(3, k)
        assert len(player.cards) == 8
    # game
    for _ in range(32):
        player_ind = board.next_player
        card = players[player_ind].get_card_to_play(board)
        points = board.add_played_card(card, verbose=verbose)
        for k, player in enumerate(players):
            player.set_reward(board, points[k % 2])
    return board


class GameBoard:
    """Elements which are visible to all players.

    Attributes
    ----------
    actions: list
        played cards, as a list of tuples of type (#player, card)
    biddings: list
        the sequence of biddings, as a list of tuples of type (#player, points, trump_suit)
    """

    def __init__(self, actions=None, biddings=None):
        self.biddings = [] if biddings is None else [(p, v, _deck._SUIT_CONVERTER.get(s, s)) for p, v, s in biddings]
        self.next_player = 0
        self.points = np.zeros((2, 32), dtype=int)
        self._actions = [] if actions is None else [(p, _deck.Card(c)) for p, c in actions]
        self._current_point_sum = 0
        self._bonus_players = set()
        self._current_point_position = 0  # checking that all cards are counted only once
        if self._actions:
            self._update_next_player()
        self._process_actions_points()

    def _as_dict(self):
        data = {"actions": [(p, c.tag) for p, c in self.actions],
                "biddings": self.biddings}
        return data

    @property
    def actions(self):
        return tuple(self._actions)  # avoid direct modification

    def dump(self, filepath):
        """Dumps a GameBoard to a file

        Parameter
        ---------
        filepath: str or Path
            path to the file where to save the GameBoard.
        """
        data = self._as_dict()
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        """Loads a GameBoard from a file

        Parameter
        ---------
        filepath: str or Path
            path to the file where the GameBoard is save.

        Returns
        -------
        GameBoard
            the loaded GameBoard
        """
        filepath = Path(filepath)
        with filepath.open("r") as f:
            data = json.load(f)
        actions = [(p, _deck.Card(c)) for p, c in data["actions"]]
        board = cls(actions, [tuple(b) for b in data["biddings"]])
        return board

    def add_played_card(self, card, verbose=False):
        """Add the next card played.
        The player is assumed to be the next_player.
        This function saves the action, updates the next player and computes points.

        Parameters
        ----------
        card: Card
            the card to play
        verbose: bool
            whether to print a summary after each round

        Returns
        -------
        np.array
            the points earned by each time, as an array of 2 elements
        """
        self._actions.append((self.next_player, card))
        player = self.next_player
        self._update_next_player()
        if verbose and not len(self._actions) % 4:
            first_player_index = self.actions[-4][0]
            print("Round #{} - Player {} starts: {}".format(len(self.actions) // 4, first_player_index,
                                                            self.get_current_round_cards().get_round_string()))
        return self._process_card_points(len(self.actions) - 1, card, player, self.next_player)

    def _update_next_player(self):
        """Updates the next_player attribute to either the following player (inside a round),
        or the winner (end of a round).
        """
        if len(self._actions) % 4:
            self.next_player = (self._actions[-1][0] + 1) % 4
        else:
            round_cards = _deck.CardList([x[1] for x in self._actions[-4:]], self.trump_suit)
            highest = round_cards.get_highest_round_card()
            index = round_cards.index(highest)
            self.next_player = (self._actions[-4][0] + index) % 4

    def _process_card_points(self, index, card, player, next_player):
        """Computes the points earned after a card being played.
        This function keeps a record of unaffected points (inside a round), and updates the "points"
        attribute.

        Returns
        -------
        np.array
            the points earned by each time, as an array of 2 elements
        """
        assert index == self._current_point_position, "Processing card #{} while expecting #{}".format(index, self._current_point_position)
        self._current_point_sum += card.get_points(self.trump_suit)
        if not (index + 1) % 4:  # end of round
            self.points[next_player % 2, index] = self._current_point_sum + (10 if index == 31 else 0)
            self._current_point_sum = 0
            # special reward
        if self.trump_suit == "❤" and card in _BONUS_CARDS:
            if player in self._bonus_players:
                self.points[player % 2, index] += 20
            self._bonus_players.add(player)
        self._current_point_position += 1
        return self.points[:, index]

    @property
    def trump_suit(self):
        """Selected trump suit for the game
        """
        return self.biddings[-1][-1]

    def __repr__(self):
        return str(self._as_dict())

    def assert_equal(self, other):
        """Asserts that the board is identical to the provided other board.
        """
        for name in ["biddings", "actions"]:
            for k, (element1, element2) in enumerate(zip(getattr(self, name), getattr(other, name))):
                if element1 != element2:
                    raise AssertionError("Discrepency with element #{} of {}: {} Vs {}".format(k, name, element1, element2))

    @property
    def is_complete(self):
        """Returns whether the game is complete
        The game is considered complete when all 32 cards are played and the 33rd element provides
        the winner of the last round.
        """
        return len(self.actions) == 32

    def assert_valid(self):
        """Asserts that the whole sequence is complete and corresponds to a valid game.
        """
        assert self.is_complete, "Game is not complete"
        assert len({x[1] for x in self.actions}) == 32, "Some cards are repeated"
        cards_by_player = list(self.replay_cards_iterator(with_trump_suit=True))
        # check the sequence
        first_player = 0
        for k, round_actions in enumerate(_utils.grouper(self.actions, 4)):
            # player order
            expected_players = (first_player + np.arange(4)) % 4
            players = [rc[0] for rc in round_actions]
            np.testing.assert_array_equal(players, expected_players, "Wrong player for round #{}".format(k))
            round_cards_list = _deck.CardList([x[1] for x in round_actions], self.trump_suit)
            first_player = (first_player + round_cards_list.index(round_cards_list.get_highest_round_card())) % 4
            # cards played
            for i, (player, card) in enumerate(round_actions):
                visible_round = _deck.CardList(round_cards_list[:i], self.trump_suit)
                error_msg = "Unauthorized {} played by player {}.".format(card, player)
                assert card in cards_by_player[player].get_playable_cards(visible_round), error_msg
                cards_by_player[player].remove(card)
        # last winner and function check
        assert first_player == self.next_player, "Wrong winner of last round"
        assert not any(x for x in cards_by_player), "Remaining cards, this function is improperly coded"

    def get_current_round_cards(self):
        """Return the cards for the current round (or the round just played if all 4 cards have been played)
        """
        end = min(len(self.actions), 32)
        start = max(0, ((end - 1) // 4)) * 4
        return _deck.CardList([x[1] for x in self.actions[start: end]], self.trump_suit)

    def _process_actions_points(self):
        """Computes the sequence of points for both teams, on a complete game.

        Returns
        -------
        np.array
            a 2x32 array, with row 0 corresponding to points earned by team #0
            (players #0 and #2) and raw 1 to team #1 (players #1 and #3) at
            each card played.

        Note
        ----
        Only the 20 point bonus can be earned out of the end of a round.
        """
        unprocessed = self.actions[self._current_point_position:]
        for k, (player, card) in enumerate(unprocessed):
            next_player = self.next_player if k + 1 == len(unprocessed) else unprocessed[k + 1][0]
            self._process_card_points(self._current_point_position, card, player, next_player)

    def replay_cards_iterator(self, with_trump_suit=False):
        """Create a new board with same card initializaton

        Parameter
        ---------
        with_trump_suit: bool
            whether to set the same trump suit to the yielded Card lists (sets it to None otherwise)

        Returns
        -------
        generator
            a generator providing the cards of each player (from a complete game)
        """
        assert self.is_complete, "Only finisehed games can be replayed"
        cards_by_player = [[] for _ in range(4)]
        for p_card in self.actions:
            cards_by_player[p_card[0]].append(p_card[1])
        return (_deck.CardList(cards, self.trump_suit if with_trump_suit else None) for cards in cards_by_player)
