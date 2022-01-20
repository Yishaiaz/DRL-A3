import itertools

import gym
import os
import shutil
import numpy as np
import tensorflow as tf2
import collections
import tensorflow.compat.v1 as tf
from Miscellaneous.utils import *

tf.disable_v2_behavior()
ENV_NAME = 'Acrobot-v1'
env = gym.make(ENV_NAME)

np.random.seed(1)

BEST_MODEL_DIR_PATH = os.path.join(os.getcwd(), 'BestModels', f"{ENV_NAME}")
BEST_MODEL_MODEL_PATH = os.path.join(BEST_MODEL_DIR_PATH, f"{ENV_NAME}_ActorCriticAdvantageBestModel")

prev_model_to_load_path = None # os.path.join(BEST_MODEL_DIR_PATH, f"{'CartPole-v1'}_ActorCriticAdvantageBestModel.meta") # or None if you don't want to load a previously trained model
SAVE_MODEL = False
LOAD_PREV_TRAINED_MODEL = False
RENDER = False
# this is where to put the environment reward where it is considered solved
# Our configuration:for cartpole=475, for acrobot=-100, mountain_car_continuous=90
GOOD_AVG_REWARD_FOR_ENV = -100

if not os.path.isdir(BEST_MODEL_DIR_PATH):
    os.makedirs(BEST_MODEL_DIR_PATH)


# Value network class
class ValueApproximationNetwork:
    def __init__(self, state_size, learning_rate, name='value_approximation_network', **kwargs):
        loss_function = kwargs.get('loss_function', tf.losses.mean_squared_error)
        network_optimizer = kwargs.get('network_optimizer', tf.train.AdamOptimizer)

        self.state_size = state_size
        self.learning_rate = learning_rate

        self.weights_initializer = tf.keras.initializers.glorot_normal(seed=0)

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.value_approximation = tf.placeholder(tf.int32, [1], name="value_approximation")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=self.weights_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1], initializer=self.weights_initializer)
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())
            # self.W3 = tf.get_variable("W3", [128, 1], initializer=self.weights_initializer)
            # self.b3 = tf.get_variable("b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            # self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            # self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.state_value_approximation = tf.squeeze(self.output)
            # Loss - type of loss function is given by the kwarg 'loss_function' and has the default
            # value of mean_squared_error function
            self.loss = tf.reduce_mean(loss_function(self.R_t, self.state_value_approximation))
            self.optimizer = network_optimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network', **kwargs):
        loss_function = kwargs.get('loss_function', tf.nn.softmax_cross_entropy_with_logits_v2)
        network_optimizer = kwargs.get('network_optimizer', tf.train.AdamOptimizer)

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.weights_initializer = tf.keras.initializers.glorot_normal(seed=0)

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=self.weights_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=self.weights_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = loss_function(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = network_optimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Metrics:
class CustomMetric(tf.keras.metrics.Metric):
    """
    custom metric to keep track of loss in each training step
    """

    def __init__(self, name='training_step_loss', dtype=tf.int32):
        super(CustomMetric, self).__init__(name=name, dtype=dtype)
        self.val = None

    def update_state(self, x):
        self.val = x

    def result(self):
        return self.val


# Define hyperparameters
state_size, action_size = get_maximum_environments_space_and_action_size()

try:
    current_env_action_size = env.action_space.n
except AttributeError:
    current_env_action_size = env.action_space.bounded_above.shape[0]

max_episodes = 2000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0004
parameters_dict = {'lr': learning_rate, 'discount': discount_factor}

# TENSORBOARD
exp_details = '_'.join([f'{key}={val}' for key, val in parameters_dict.items()])
exp_details = f"{ENV_NAME}_{'using_pre_trained_' if prev_model_to_load_path is not None else ''}ActorCritic_{exp_details}"

main_dir_to_save = os.sep.join([os.getcwd(), 'Experiments'])
exp_dir_to_save_train = os.sep.join([main_dir_to_save, exp_details, 'train'])
exp_dir_to_save_test = os.sep.join([main_dir_to_save, exp_details, 'test'])
# remove all existing dirs and files with the same experiment identifier.
if os.path.isdir(exp_dir_to_save_train):
    shutil.rmtree(exp_dir_to_save_train, ignore_errors=True)
if os.path.isdir(exp_dir_to_save_test):
    shutil.rmtree(exp_dir_to_save_test, ignore_errors=True)


# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)
# ADDED
value_approximation_network = ValueApproximationNetwork(state_size=state_size, learning_rate=0.02)
# /ADDED

# TENSORBOARD
tf.compat.v1.summary.scalar(name="episode_reward", tensor=policy.R_t)
tf.compat.v1.summary.scalar(name="episode_learning_rate", tensor=policy.learning_rate)
tf.compat.v1.summary.scalar(name="episode_loss", tensor=policy.loss)

tfb_train_summary_writer = tf.summary.FileWriter(exp_dir_to_save_train)
summaries = tf.compat.v1.summary.merge_all()
# / TENSORBOARD

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done",
                                                       'value_approximation_of_state'])
    saver = tf.train.Saver()
    # load the model if a trained one already exists:
    if LOAD_PREV_TRAINED_MODEL and os.path.isfile(f"{BEST_MODEL_MODEL_PATH}.meta"):
        best_model_path = f"{BEST_MODEL_MODEL_PATH}.meta" if prev_model_to_load_path is None else prev_model_to_load_path
        loader = tf.compat.v1.train.import_meta_graph(f"{BEST_MODEL_MODEL_PATH}.meta")
        loader.restore(sess, BEST_MODEL_MODEL_PATH)
        LOAD_PREV_TRAINED_MODEL = True

    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    train = True
    for episode in range(max_episodes):
        state = env.reset()
        state = fill_space_vector_with_zeros(state, state_size)
        state = state.reshape([1, state_size])
        episode_transitions = []
        i = 1
        for step in range(max_steps):
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            actions_distribution = actions_distribution[: current_env_action_size]
            actions_distribution = actions_distribution/actions_distribution.sum()
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = fill_space_vector_with_zeros(next_state, state_size)
            next_state = next_state.reshape([1, state_size])

            if RENDER:
                env.render()

            value_approximation_of_curr_state = sess.run(value_approximation_network.state_value_approximation,
                                                         {value_approximation_network.state: state})

            value_approximation_of_next_state = sess.run(value_approximation_network.state_value_approximation,
                                                         {value_approximation_network.state: next_state})

            target = reward + discount_factor * value_approximation_of_next_state * (int(1 - done))
            delta = target - value_approximation_of_curr_state

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1

            # replaced discounted return with the state advantage

            if (train):
                policy_net_feed_dict = {policy.state: state, policy.R_t: delta, policy.action: action_one_hot}
                # policy network update
                _, policy_net_loss = sess.run([policy.optimizer, policy.loss], policy_net_feed_dict)

                # updating the value approximation network as well
                value_approximation_net_feed_dict = {value_approximation_network.state: state,
                                                     value_approximation_network.R_t: target}

                _, value_net_loss = sess.run(
                    [value_approximation_network.optimizer, value_approximation_network.loss],
                    value_approximation_net_feed_dict)

            episode_rewards[episode] += reward
            i = discount_factor * i

            if done:
                policy_net_feed_dict[policy.R_t] = episode_rewards[episode]
                summary = sess.run(summaries, policy_net_feed_dict)
                tfb_train_summary_writer.add_summary(summary, episode)
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    if average_rewards > GOOD_AVG_REWARD_FOR_ENV:
                        print(' Solved at episode: ' + str(episode))
                        solved = True

                print(
                    "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                 round(average_rewards, 2)))

                if (episode_rewards[episode] > GOOD_AVG_REWARD_FOR_ENV):
                    train = False
                else:
                    train = True
                break
            state = next_state

        if solved:
            # save model
            if SAVE_MODEL:
                saver.save(sess, BEST_MODEL_MODEL_PATH)
            break

