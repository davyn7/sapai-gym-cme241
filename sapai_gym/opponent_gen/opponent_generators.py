from sapai import Player, Team
from sapai_gym.ai import baselines
from sapai_gym import SuperAutoPetsEnv
from stable_baselines3 import DQN

# TODO : Wrap the ai to create a generator


def _do_store_phase(env: SuperAutoPetsEnv, ai):
    env.player.start_turn()
    if ai == "model_agent":
        while True:
            actions = env._avail_actions()
            obs = env.get_scaled_state()
            encoded = obs
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            model = DQN.load("Directory to model here")
            action, new_state = model.predict(encoded_reshaped, verbose=0).flatten()
            chosen_action = action[0]
            env.resolve_action(chosen_action)
            if SuperAutoPetsEnv._get_action_name(actions[chosen_action]) == "end_turn":
                return



    while True:
        actions = env._avail_actions()
        chosen_action = ai(env.player, actions)
        env.resolve_action(chosen_action)

        if SuperAutoPetsEnv._get_action_name(actions[chosen_action]) == "end_turn":
            return


def opp_generator(num_turns, ai):
    opps = list()
    env = SuperAutoPetsEnv(None, valid_actions_only=True, manual_battles=True)
    while env.player.turn <= num_turns:
        _do_store_phase(env, ai)
        opps.append(Team.from_state(env.player.team.state))
    return opps


def random_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.random_agent)


def biggest_numbers_horizontal_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.biggest_numbers_horizontal_scaling_agent)

def model_opp_generator(num_turns):
    return opp_generator(num_turns, ai="model_agent")