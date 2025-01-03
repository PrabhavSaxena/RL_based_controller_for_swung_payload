from swung_env import *
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

drone = DRONE_ENV(gain_pos_x0=[0, 0, 0], gain_pos_y0=[0, 0, 0],
                  gain_thrust0=[0, 0, 0], gain_roll0=[0, 0, 0], gain_pitch0=[0, 0, 0],
                  time_per_cycle=0.5)

drone.s = [0, 0, 0, 0, 0, 0]
a = [0, 0, 0]
s = 0

'''
while time.time() - drone.start < 50:
    drone.run(action(a))
    if drone.t % float(drone.time_per_cycle) < 0.01:
        s, a, s_curr = (drone.s, drone.a, drone.s_new)

    print(s, a, drone.s_new)
'''

num_states = 6
num_actions = 2
lower_bound = 0
upper_bound = 2.5


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        # self.action_buffer[index] = obs_tuple[1]
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out_1 = layers.Dense(256, activation="relu")(inputs)
    out_2 = layers.Dense(256, activation="relu")(out_1)
    out_3 = layers.Dense(256, activation="relu")(out_2)
    out = layers.Dense(256, activation="relu")(out_3)
    outputs = layers.Dense(num_actions, activation="sigmoid", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out_1 = layers.Dense(16, activation="relu")(state_input)
    state_out_2 = layers.Dense(16, activation="relu")(state_out_1)
    state_out_3 = layers.Dense(16, activation="relu")(state_out_2)
    state_out = layers.Dense(32, activation="relu")(state_out_3)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out_1 = layers.Dense(256, activation="relu")(out)
    out_2 = layers.Dense(256, activation="relu")(out_1)
    out_3 = layers.Dense(256, activation="relu")(out_2)
    outputs = layers.Dense(1)(out_3)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
# actor_model.load_weights('actor_model_max_reward_30s10.h5')
# critic_model.load_weights('critic_model_max_reward_30s10.h5')
# target_actor.load_weights('actor_model_max_reward_30lr_001.h5')
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)


def step(action, prev_state):
    drone.gain_thrust0[0] = action[0][0]
    drone.gain_thrust0[1] = action[0][1]
    lst = drone.d.qpos[:5]
    lst = np.append(lst, drone.last_errors0[2])
    state = np.array(lst)
    reward = (-abs(state[2] - 5) + 5) # (4*abs(drone.error) + 0.3*abs(drone.diff_error))  # + 10*(-abs(prev_state[5] - state[5]) + 5)
    return state, reward, False, False


ep_reward_list = []
avg_reward_list = []
action = [0, 0, 0]
prev_state = drone.reset()
reward = 0
max_reward = float('-inf')
max_reward_episode = 0
for ep in range(total_episodes):
    t = 0
    time.sleep(2)
    start_time = time.time()
    prev_state = drone.reset()
    state = [0, 0, 0, 0, 0, 0]
    episodic_reward = 0
    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
        while viewer.is_running() and t <= 30:
            step_start = time.time()
            t = time.time() - start_time
            prev_state = drone.return_state[0]
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # print(prev_state)
            if t % 0.1 < 0.01:
                # prev_state = s_curr
                # print(drone.return_state[0][2])
                # print(tf_prev_state)
                action = policy(tf_prev_state, ou_noise)
                state, reward, done, info = step(action, prev_state)
                episodic_reward += reward
                # drone.gain_thrust0 = action[0]
                # # print(time.time() - drone.start)
                # # print(action)
                # reward = drone.get_reward()
                print("##############################################")
                print("REWARD: ", reward)
                print("ACTION: ", action)
                print("##############################################")
                # drone.s_last, drone.s_curr = drone.return_state()

            control = drone.controller()

            mujoco.set_mjcb_control(drone.action_motor(control))
            mujoco.mj_step(drone.m, drone.d)

            # _, reward, _, done = step(prev_state, s_curr)
            # print(" Action: ", action, " PREV_STATE: ", drone.s_last[:3], " STATE: ", drone.d.qpos[:3])
            # done = False
            # print(reward)
            # Receive state and reward from environment.

            buffer.record((prev_state, action, reward, state))

            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            prev_state = state
            viewer.sync()

    ep_reward_list.append(episodic_reward)

    if episodic_reward > max_reward:
        max_reward = episodic_reward
        max_reward_episode = ep
        actor_model.save_weights("actor_model_max_reward_30pd1.h5")
        critic_model.save_weights("critic_model_max_reward_30pd1.h5")
    print("Actor model weights saved for episode with max reward:", max_reward_episode)

    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, ep_reward_list))
    avg_reward_list.append(avg_reward)

    # # Mean of last 40 episodes
    # if max_reward_episode > 0:
    #     actor_model.save_weights("actor_model_max_reward_30s2.h5")
    #     print("Actor model weights saved for episode with max reward:", max_reward_episode)

# Plotting graph
# Episodes versus Avg. Rewards
print(ep_reward_list)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.show()
