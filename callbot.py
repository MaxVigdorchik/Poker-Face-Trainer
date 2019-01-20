from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

import pickle
import numpy as np
import pymc3 as pm
import logging
logger = logging.getLogger("pymc3")
logger.propagate = False

simulation_num = 1000


class CallBot(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action_model = pm.Model()
        community = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=simulation_num,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community)
        )
        print("Win Rate Is ", win_rate)

        with action_model:
            win_chance = pm.Normal('win_chance', mu=win_rate, sd=np.sqrt(
                win_rate*(1-win_rate)/simulation_num))
            # TODO: Standard deviations for confidence can be calculated in a smarter way
            confidences = [
                pm.Normal('confidence_' + id, mu=self.round_players[id]['confidence'][0], sd=0.2) for id in self.game_uuids]
            mu = pm.Normal('base', mu=1, sd=0.3)
            bluffs = [pm.Normal('bluff_' + id, mu=mu, sd=0.3,
                                observed=self.game_players[id]['bluffs']) for id in self.game_uuids]

            trace = pm.sample(500, progressbar=False)
            post_pred = pm.sample_posterior_predictive(
                trace, samples=500, progressbar=False)
            print(np.mean(post_pred['bluff_'+self.game_uuids[0]]))

        actions = [item for item in valid_actions if item['action'] in ['call']]
        return list(np.random.choice(actions).values())

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        self.game_uuids = []
        game_info_copy = game_info['seats'].copy()
        self.game_players = {}
        for p in game_info_copy:
            if not p['uuid'] == self.uuid:
                self.game_players[p['uuid']] = p
                self.game_players[p['uuid']]['average_confidence'] = (
                    0.5, 1)  # Average is (sum,amount), initialize nonzero to reduce variance
                self.game_players[p['uuid']]['average_loss_confidence'] = (
                    0.5, 1)
                self.game_players[p['uuid']]['average_win_confidence'] = (
                    0.5, 1)

                # List of relative confidence after losses
                self.game_players[p['uuid']]['bluffs'] = []
                self.game_uuids.append(p['uuid'])
        self.load_data()

    def receive_round_start_message(self, round_count, hole_card, seats):
        round_players_copy = seats.copy()
        self.round_players = {}
        for p in round_players_copy:
            if not p['uuid'] == self.uuid:
                self.round_players[p['uuid']] = p
                self.round_players[p['uuid']]['actions'] = []
                self.round_players[p['uuid']]['confidence'] = (0, 1)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        uuid = action['player_uuid']
        if not uuid == self.uuid:
            self.round_players[uuid]['actions'].append(
                (action['action'], action['amount'], round_state['pot']['main']['amount']))
            # TODO: This ignores sidepot, which could be very important if bottom stack is all in

            confidence = 0
            for action in self.round_players[action['player_uuid']]['actions']:
                if action[0] == 'fold':
                    confidence = 0
                    break
                confidence += action[1]/action[2]  # Amount divided by pot size
            # Keep track of action number as well
            self.round_players[uuid]['confidence'] = (
                confidence, len(self.round_players[uuid]['actions']))

    def receive_round_result_message(self, winners, hand_info, round_state):
        win_ids = [p['uuid'] for p in winners]
        if hand_info:
            for uuid in self.game_uuids:
                # Update average confidences FOR FINISHED HANDS
                total, number = self.game_players[uuid]['average_confidence']
                self.game_players[uuid]['average_confidence'] = (
                    total + self.round_players[uuid]['confidence'][0], number + 1)

                if uuid in win_ids:
                    total, number = self.game_players[uuid]['average_win_confidence']
                    self.game_players[uuid]['average_win_confidence'] = (
                        total + self.round_players[uuid]['confidence'][0], number + 1)
                else:
                    total, number = self.game_players[uuid]['average_loss_confidence']
                    self.game_players[uuid]['average_loss_confidence'] = (
                        total + self.round_players[uuid]['confidence'][0], number + 1)

                    self.game_players[uuid]['bluffs'].append(
                        self.round_players[uuid]['confidence'][0]/(total / number))
        print(self.game_players[self.game_uuids[0]]['bluffs'])
        self.save_data()

    def save_data(self):
        with open('player.dat', 'rb') as f:
            players = pickle.load(f)

        for uuid in self.game_uuids:
            players[self.game_players[uuid]['name']] = (self.game_players[uuid]['bluffs'], self.game_players[uuid]['average_confidence'],
                                                        self.game_players[uuid]['average_loss_confidence'], self.game_players[uuid]['average_win_confidence'])

        with open('player.dat', 'wb') as f:
            players = pickle.dump(players, f)

    def load_data(self):
        with open('player.dat', 'rb') as f:
            players = pickle.load(f)

        for uuid in self.game_uuids:
            if self.game_players[uuid]['name'] in players.keys():
                data = players[self.game_players[uuid]['name']]
                self.game_players[uuid]['bluffs'] = data[0]
                self.game_players[uuid]['average_confidence'] = data[1]
                self.game_players[uuid]['average_loss_confidence'] = data[2]
                self.game_players[uuid]['average_win_confidence'] = data[3]


def setup_ai():
    return CallBot()
