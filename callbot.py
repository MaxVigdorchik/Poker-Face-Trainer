from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt

simulation_num = 1000
basic_model = pm.Model()


def sample_wins(win_rate):
    with basic_model:
        a = pm.Normal('alpha', mu=win_rate, sigma=0.2)


class CallBot(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        community = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=simulation_num,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community)
        )
        print(win_rate)
        actions = [item for item in valid_actions if item['action'] in ['call']]
        return list(np.random.choice(actions).values())

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return CallBot()
