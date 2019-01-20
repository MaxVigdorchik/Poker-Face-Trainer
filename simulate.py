from pypokerengine.api.game import start_poker, setup_config

from callbot import CallBot
from databloggerbot import DataBloggerBot
import numpy as np

if __name__ == '__main__':
    blogger_bot = DataBloggerBot()

    # The stack log contains the stacks of the Data Blogger bot after each game (the initial stack is 100)
    stack_log = []
    for round in range(50):
        p1, p2 = blogger_bot, CallBot()

        config = setup_config(
            max_round=50, initial_stack=100, small_blind_amount=5)
        config.register_player(name="blogger", algorithm=p1)
        config.register_player(name="superbot", algorithm=p2)
        game_result = start_poker(config, verbose=0)

        stack_log.append(
            [player['stack'] for player in game_result['players'] if player['uuid'] == p2.uuid])
        print('Avg. stack:', '%d' % (int(np.mean(stack_log))))
