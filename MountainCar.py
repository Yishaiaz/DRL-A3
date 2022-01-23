import gym
import os
import shutil
import numpy as np
import tensorflow as tf
from Miscellaneous.utils import *
from ActorCriticAgent import *
from datetime import datetime
from tensorflow.keras import metrics
import sklearn
import sklearn.preprocessing

ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME)

np.random.seed(65483)

SAVE_MODEL = True
LOAD_PREV_OTHER_MODEL = False
LOAD_PREV_MODEL = False
BASE_MODEL_NAME = 'CartPole-v1'
RENDER = False
# this is where to put the environment reward where it is considered solved
# Our configuration:for cartpole=475, for acrobot=-100, mountain_car_continuous=90
GOOD_AVG_REWARD_FOR_ENV = 90
    
# Define hyperparameters
state_size, action_size = get_maximum_environments_space_and_action_size()
try:
    current_env_action_size = env.action_space.n
except AttributeError:
    current_env_action_size = env.action_space.bounded_above.shape[0]
                                    
state_space_samples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

#function to normalize states
def scale_state(state):                 #requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled[0]                     

max_episodes = 1000
max_steps = 1000
discount_factor = 0.999


actor = ActorDist(state_size,[64,32],-1,1,ENV_NAME,0.0001)

critic = Critic(state_size,[64,32],ENV_NAME,0.0001)

if LOAD_PREV_MODEL:
    actor.load_model()
    actor.freeze_train()
    critic.load_model()

if LOAD_PREV_OTHER_MODEL:
    actor.load_base('./weights/'+BASE_MODEL_NAME+'_actor.h5')
    critic.load_base('./weights/'+BASE_MODEL_NAME+'_critic.h5')


critic_loss_metric = metrics.Mean('critic_loss', dtype=tf.float32)
actor_loss_metric = metrics.Mean('actor_loss', dtype=tf.float32)


current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/'+ENV_NAME + '/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

def learn(state, action, reward, next_state, done):
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        action,norm_dist = actor.take_action(state)
        value_approximation_of_curr_state = critic.predict(state)

        value_approximation_of_next_state = critic.predict(next_state)

        target = reward + discount_factor * value_approximation_of_next_state * (int(1 - done))
        delta = target - value_approximation_of_curr_state

        actor_loss = actor.loss(action,norm_dist,delta)
        actor_loss_metric(actor_loss)
        critic_loss = critic.loss(target,value_approximation_of_curr_state)
        critic_loss_metric(critic_loss)

    grads1 = tape1.gradient(actor_loss, actor.model.trainable_variables)
    grads2 = tape2.gradient(critic_loss, critic.model.trainable_variables)
    actor.optimizer.apply_gradients(zip(grads1, actor.model.trainable_variables))
    critic.optimizer.apply_gradients(zip(grads2, critic.model.trainable_variables))

solved = False
episode_rewards = np.zeros(max_episodes)
episode_rewards_p = np.zeros(max_episodes)
average_rewards = 0.0
train = True

for episode in range(max_episodes):
        
        state = env.reset()
        state = fill_space_vector_with_zeros(scale_state(state), state_size)
        state = state.reshape([1, state_size])
        episode_transitions = []
        i = 1
        actor_loss_metric.reset_states()
        critic_loss_metric.reset_states()
        for step in range(max_steps):
            action,_ = actor.take_action(state)
            next_state, reward, done, _ = env.step(action) 
            next_state = scale_state(next_state)
                                    
            next_state = fill_space_vector_with_zeros(next_state, state_size)
            next_state = next_state.reshape([1, state_size])

            if RENDER:
                env.render()
                
            if (train):
                learn(state, action, reward, next_state, done)
            episode_rewards[episode] += reward

            if done or step == max_steps-1:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    
                print(
                    "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                 round(average_rewards, 2)))
                with summary_writer.as_default():
                    tf.summary.scalar('actor_loss', actor_loss_metric.result(), step=episode)
                    tf.summary.scalar('critic_loss', critic_loss_metric.result(), step=episode)
                    tf.summary.scalar('avg_reward', average_rewards, step=episode)
                    tf.summary.scalar('reward', episode_rewards[episode], step=episode)
                    
                if (average_rewards > GOOD_AVG_REWARD_FOR_ENV and episode > 98):
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                if (episode_rewards[episode] > GOOD_AVG_REWARD_FOR_ENV):
                    train = False
                else:
                    train = True
                break
            state = next_state

        if solved:
            actor.save_base()
            critic.save_base()
            break


