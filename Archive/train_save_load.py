# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:25:11 2022

@author: qbr5kx
"""

def train(self, stop_criteria):
    """
    Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
    :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
    :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
    """
    analysis = ray.tune.run(ppo.PPOTrainer, config=self.config, local_dir=self.save_dir, stop=stop_criteria,
                            checkpoint_at_end=True)
    # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
    checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean'),
                                                       metric='episode_reward_mean')
    # retriev the checkpoint path; we only have a single checkpoint, so take the first one
    checkpoint_path = checkpoints[0][0]
    return checkpoint_path, analysis

def load(self, path):
    """
    Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
    :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    """
    self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
    self.agent.restore(path)

def test(self):
    """Test trained agent for a single episode. Return the episode reward"""
    # instantiate env class
    env = self.env_class(self.env_config)

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = self.agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward