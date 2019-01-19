from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


import numpy as np
import theano.tensor as tt
import pymc3 as pm

simulation_num = 1000


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

        for uuid, p in self.round_players:
            confidence = 0
            for action in p['actions']:
                if action[0] == 'fold':
                    confidence = 0
                    break
                confidence += action[1]/action[2]  # Amount divided by pot size

        actions = [item for item in valid_actions if item['action'] in ['call']]
        return list(np.random.choice(actions).values())

    def receive_game_start_message(self, game_info):
        print(game_info)
        self.nb_player = game_info['player_num']
        game_info_copy = game_info['seats'].copy()
        self.game_players = {}
        for p in game_info_copy:
            if not p['uuid'] == self.uuid:
                self.game_players[p['uuid']] = p

    def receive_round_start_message(self, round_count, hole_card, seats):
        round_players_copy = seats.copy()
        self.round_players = {}
        for p in round_players_copy:
            if not p['uuid'] == self.uuid:
                self.round_players[p['uuid']] = p
                self.round_players[p['uuid']]['actions'] = []

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        if not action['player_uuid'] == self.uuid:
            self.round_players[action['player_uuid']]['actions'].append(
                (action['action'], action['amount'], round_state['pot']['main']['amount']))
            # TODO: This ignores sidepot, which could be very important if bottom stack is all in

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(winners)
        pass


def setup_ai():
    return CallBot()
