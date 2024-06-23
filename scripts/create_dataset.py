# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import os
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

import sys, os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environments.env_policies as env_policies
import data.tf_agents_onpolicy_dataset as tf_agents_onpolicy_dataset
import estimators.estimator as estimator_lib
import utils.common as common_utils
from data.dataset import Dataset, EnvStep, StepType
from data.tf_offpolicy_dataset import TFOffpolicyDataset

from gym.envs.toy_text.frozen_lake import generate_random_map


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'taxi', 'Environment name.')
flags.DEFINE_integer('env_size', 4, 'Dimensions of the environment (Default is 4x4)')
flags.DEFINE_bool('default', False, 'Whether the default map of the environment is used')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 100, 'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 500, 'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 1.0, 'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('save_dir', None, 'Directory to save dataset to.')
flags.DEFINE_string('load_dir', None, 'Directory to load policies from.')
flags.DEFINE_bool('force', False, 'Whether to force overwriting any existing dataset.')


def get_onpolicy_dataset(load_dir, env_name, tabular_obs, max_trajectory_length, alpha, seed, gym_kwargs):
  """
  Prepares an on-policy dataset.
  - It loads the environment and policy using env_policies.get_env_and_policy().
  - Then it creates a dataset using tf_agents_onpolicy_dataset.TFAgentsOnpolicyDataset().
  """
  tf_env, tf_policy = env_policies.get_env_and_policy(load_dir, env_name, alpha, env_seed=seed, tabular_obs=tabular_obs, gym_kwargs=gym_kwargs)

  dataset = tf_agents_onpolicy_dataset.TFAgentsOnpolicyDataset(tf_env, tf_policy, episode_step_limit=max_trajectory_length)
  return dataset


def add_episodes_to_dataset(episodes, valid_ids, write_dataset):
  """
  Adds episodes to the dataset.
  - It iterates through the episodes and steps to add them to the dataset.
  """
  num_episodes = 1 if tf.rank(valid_ids) == 1 else tf.shape(valid_ids)[0]
  for ep_id in range(num_episodes):
    if tf.rank(valid_ids) == 1:
      this_valid_ids = valid_ids
      this_episode = episodes
    else:
      this_valid_ids = valid_ids[ep_id, ...]
      this_episode = tf.nest.map_structure(
          lambda t: t[ep_id, ...], episodes)

    episode_length = tf.shape(this_valid_ids)[0]
    for step_id in range(episode_length):
      this_valid_id = this_valid_ids[step_id]
      this_step = tf.nest.map_structure(
          lambda t: t[step_id, ...], this_episode)
      if this_valid_id:
        write_dataset.add_step(this_step)


def main(argv):
  env_name = FLAGS.env_name
  env_size = FLAGS.env_size
  default = FLAGS.default
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  save_dir = FLAGS.save_dir
  load_dir = FLAGS.load_dir
  force = FLAGS.force

  gym_kwargs = None
  # Generates a random map for the 'Frozen Lake' environment of size 'env_size'
  if not default:
    gym_kwargs = { 'desc': generate_random_map(size=env_size) }

  # Generates a new directory to save the dataset
  hparam_str = ('{ENV_NAME}_size{ENV_SIZE}_default{DEFAULT}_tabular{TAB}_'
                'alpha{ALPHA}_seed{SEED}_numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    ENV_SIZE=env_size,
                    DEFAULT=default,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(save_dir, hparam_str)
  # Checks if the directory already exists
  if tf.io.gfile.isdir(directory) and not force:
    raise ValueError('Directory %s already exists. Use --force to overwrite.' %directory)

  # Sets random seed
  np.random.seed(seed)
  tf.random.set_seed(seed)

  dataset = get_onpolicy_dataset(load_dir, env_name, tabular_obs, max_trajectory_length, alpha, seed, gym_kwargs)
  write_dataset = TFOffpolicyDataset(dataset.spec, capacity=num_trajectory * (max_trajectory_length + 1))  # Creates a dataset for writing
  
  batch_size = 20

  state_distribution_behaviour = {}  # Dictionary with the state-action pairs and their probability
  state_action_pair_counts = {}  # Empty dictionary to store the state-action pairs and their occurrences
  total_state_action_pairs = 0

  # Iterates over the batches of trajectories
  for batch_num in range(1 + (num_trajectory - 1) // batch_size):
    num_trajectory_after_batch = min(num_trajectory, batch_size * (batch_num + 1))
    num_trajectory_to_get = num_trajectory_after_batch - batch_num * batch_size
    episodes, valid_steps = dataset.get_episode(batch_size=num_trajectory_to_get)  # Gets a batch of trajectories from the dataset
    
    batch_states = episodes.observation.numpy()  # Converts the batch of observations into a NumPy array since Tensor is unhashable
    batch_actions = episodes.action.numpy()  # Converts the batch of actions into a NumPy array since Tensor is unhashable

    # Iterates over each episode and each step within the episode
    for episode in range(len(batch_actions)):
      for step in range(len(batch_actions[episode])):
        state = batch_states[episode][step]
        state_action_pair = (state, batch_actions[episode][step])  # Creates a tuple of the state and action
        state_action_pair_counts[state_action_pair] = state_action_pair_counts.get(state_action_pair, 0) + 1  # Increments the occurrence count
        total_state_action_pairs += 1
    
    # Calculates the state-visitation distribution
    for (state, action), count in state_action_pair_counts.items():
      state_distribution_behaviour[(state, action)] = count / total_state_action_pairs
    
    add_episodes_to_dataset(episodes, valid_steps, write_dataset)

    print('num episodes collected: %d', write_dataset.num_total_episodes)
    print('num steps collected: %d', write_dataset.num_steps)

    estimate = estimator_lib.get_fullbatch_average(write_dataset)
    print('per step avg on off-policy data', estimate)
    estimate = estimator_lib.get_fullbatch_average(write_dataset,by_steps=False)
    print('per episode avg on off-policy data', estimate)

  # Generates a new file name to save the state-visitation distribution
  file_name = ('3a_state_visitation_distributions_{ENV_NAME}_size{ENV_SIZE}_default{DEFAULT}_tabular{TAB}_'
               'alpha{ALPHA}_seed{SEED}_numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    ENV_SIZE=env_size,
                    DEFAULT=default,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  # Saves the state-visitation distribution to a file
  with open(os.path.join('./state_visitation_mismatch', file_name), 'wb') as f:
    pickle.dump(state_distribution_behaviour, f)

  print('Saving dataset to %s.' % directory)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)
  write_dataset.save(directory)

  print('Loading dataset.')
  new_dataset = Dataset.load(directory)
  print('num loaded steps', new_dataset.num_steps)
  print('num loaded total steps', new_dataset.num_total_steps)
  print('num loaded episodes', new_dataset.num_episodes)
  print('num loaded total episodes', new_dataset.num_total_episodes)

  estimate = estimator_lib.get_fullbatch_average(new_dataset)
  print('per step avg on saved and loaded off-policy data', estimate)
  estimate = estimator_lib.get_fullbatch_average(new_dataset, by_steps=False)
  print('per episode avg on saved and loaded off-policy data', estimate)

  print('Done!')


if __name__ == '__main__':
  app.run(main)
