# -*- coding: utf-8 -*
from pathlib import Path
import ujson as json
import numpy as np
from . import _deck
from . import _utils


class DefaultPlayer:
    """Player which selects one card randomly at each round
    """

    def __init__(self):
        self._cards = _deck.CardList([])
        self._order = None

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
        self._order = order

    def get_card_to_play(self, board):
        if self._cards.trump_suit is None:
            self._cards.trump_suit = board.trump_suit
        round_cards = board.get_current_round_cards()
        playable = self._cards.get_playable_cards([] if len(round_cards) == 4 else round_cards)
        selected = np.random.choice(playable)
        self._cards.remove(selected)
        return selected


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
    cards = [_deck.Card(v + s) for s in _deck.SUITS for v in _deck.VALUES]
    np.random.shuffle(cards)
    for k, cards in enumerate(_utils.grouper(cards, 8)):
        players[k].initialize_game(k, _deck.CardList(cards))


def play_game(board, players, verbose=False):
    """Plays a game, given a board with biddings and initialized players.

    Parameters
    ----------
    board: GameBoard
        a board, with biddings already performed, and no played cards
    players: list
        a list of 4 initialized players, with 8 cards each and given orders
    """  # IMPROVEMENT: handle partially played games
    # checks
    assert board.biddings, "Biddings must have been already performed"
    assert not board.played_cards, "No cards should have already been played"
    for k, player in enumerate(players):  # make sure the data is correct
        assert player._order == min(3, k)
        assert len(player.cards) == 8
    # game
    for _ in range(32):
        player_ind = board.next_player
        card = players[player_ind].get_card_to_play(board)
        board.add_played_card(card, verbose=verbose)
    return board


class GameBoard:
    """Elements which are visible to all players.

    Attributes
    ----------
    played_cards: list
        played cards, as a list of tuples of type (#player, card)
    biddings: list
        the sequence of biddings, as a list of tuples of type (#player, points, trump_suit)
    """

    def __init__(self, played_cards=None, biddings=None):
        self.played_cards = [] if played_cards is None else played_cards
        self.biddings = [] if biddings is None else biddings
        self.next_player = 0

    def _as_dict(self):
        data = {"played_cards": [(p, c.tag) for p, c in self.played_cards],
                "biddings": self.biddings,
                "next_player": self.next_player}
        return data

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
        played_cards = [(p, _deck.Card(c) if c != 'None' else None) for p, c in data["played_cards"]]
        board = cls(played_cards, [tuple(b) for b in data["biddings"]])
        board.next_player = data["next_player"]
        return board

    def add_played_card(self, card, verbose=False):
        self.played_cards.append((self.next_player, card))
        if len(self.played_cards) % 4:
            self.next_player = (self.next_player + 1) % 4
        else:
            round_cards = _deck.CardList([x[1] for x in self.played_cards[-4:]], self.trump_suit)
            highest = round_cards.get_highest_round_card()
            index = round_cards.index(highest)
            self.next_player = (self.next_player + 1 + index) % 4
            if verbose:
                first_player_index = self.played_cards[-4][0]
                print("Round #{} - Player {} starts: {}".format(len(self.played_cards) // 4, first_player_index,
                                                                round_cards.get_round_string()))

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
        for name in ["biddings", "played_cards"]:
            for k, (element1, element2) in enumerate(zip(getattr(self, name), getattr(other, name))):
                if element1 != element2:
                    raise AssertionError("Discrepency with element #{} of {}: {} Vs {}".format(k, name, element1, element2))

    @property
    def is_complete(self):
        """Returns whether the game is complete
        The game is considered complete when all 32 cards are played and the 33rd element provides
        the winner of the last round.
        """
        return len(self.played_cards) == 32

    def assert_valid(self):
        """Asserts that the whole sequence is complete and corresponds to a valid game.
        """
        assert self.is_complete, "Game is not complete"
        assert len({x[1] for x in self.played_cards}) == 32, "Some cards are repeated"
        cards_by_player = [[] for _ in range(4)]
        for p_card in self.played_cards:
            cards_by_player[p_card[0]].append(p_card[1])
        cards_by_player = [_deck.CardList(c, self.trump_suit) for c in cards_by_player]
        # check the sequence
        first_player = 0
        for k, round_played_cards in enumerate(_utils.grouper(self.played_cards, 4)):
            # player order
            expected_players = (first_player + np.arange(4)) % 4
            players = [rc[0] for rc in round_played_cards]
            np.testing.assert_array_equal(players, expected_players, "Wrong player for round #{}".format(k))
            round_cards_list = _deck.CardList([x[1] for x in round_played_cards], self.trump_suit)
            first_player = (first_player + round_cards_list.index(round_cards_list.get_highest_round_card())) % 4
            # cards played
            for i, (player, card) in enumerate(round_played_cards):
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
        end = min(len(self.played_cards), 32)
        start = max(0, ((end - 1) // 4)) * 4
        return _deck.CardList([x[1] for x in self.played_cards[start: end]], self.trump_suit)

    def compute_points(self):
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
        assert self.is_complete, "Cannot compute the whole sequence on the fly"
        points = np.zeros((2, 32))
        special_cards = {_deck.Card(c) for c in ["Qh", "Kh"]}
        special_card_players = set()
        trump_suit = self.trump_suit
        current_sum = 0
        for k, (player, card) in enumerate(self.played_cards[:32]):
            current_sum += card.get_points(trump_suit)
            if not (k + 1) % 4:
                if k == 31:
                    points[self.next_player % 2, k] = current_sum + 10
                else:
                    winner = self.played_cards[k + 1][0]
                    points[winner % 2, k] = current_sum + (10 if k == 31 else 0)
                current_sum = 0
            # special reward
            if card in special_cards:
                if player in special_card_players:
                    points[player % 2, k] += 20
                special_card_players.add(player)
        return points
