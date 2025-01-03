from swung_env import *
import tensorflow as tf
from tensorflow.keras import layers

drone = DRONE_ENV(gain_pos_x0=[0, 0, 0], gain_pos_y0=[0, 0, 0],
                  gain_thrust0=[0, 0, 0], gain_roll0=[0, 0, 0], gain_pitch0=[0, 0, 0],
                  time_per_cycle=0.5)

drone.s = [0, 0, 0, 0, 0, 0]
a = [0, 0, 0]
s = 0

num_states = 6
num_actions = 3
lower_bound = 0
upper_bound = 5


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="sigmoid", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


# Assuming you've already defined the get_actor function
loaded_actor_model = get_actor()
loaded_actor_model.load_weights("actor_model_max_reward_30s6.h5")
# loaded_actor_model.summary()

with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
    t = 0
    start_time = time.time()
    while viewer.is_running() and t <= 1500:
        step_start = time.time()
        t = time.time() - start_time
        prev_state = drone.return_state[0]
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        if t % 0.1 < 0.01:
            # prev_state = s_curr
            input_data = tf.convert_to_tensor(
                drone.return_state[0])  # Adjust 'input_shape' based on your model's input shape
            input_data = input_data[tf.newaxis, :]
            # print(input_data)
            # Make predictions using the loaded model
            predictions = loaded_actor_model.predict(input_data)
            drone.gain_thrust0[0] = predictions[0][0]
            drone.gain_thrust0[1] = predictions[0][1]
            drone.gain_thrust0[2] = predictions[0][2]

            drone.update_pos()
            # print(drone.gain_thrust0)
            # print(time.time() - drone.start)
            # print(action)
            # drone.s_last, drone.s_curr = drone.return_state()

        control = drone.controller()
        print(drone.s_curr[2])
        mujoco.set_mjcb_control(drone.action_motor(control))
        mujoco.mj_step(drone.m, drone.d)

        # _, reward, _, done = step(prev_state, s_curr)
        # print(" Action: ", action, " PREV_STATE: ", drone.s_last[:3], " STATE: ", drone.d.qpos[:3])
        done = False
        # print(reward)
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(drone.d.time % 2)

        time_until_next_step = drone.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # Receive state and reward from environment.

        viewer.sync()

drone.logger_distance()