import math
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time


class DRONE_ENV:

    def __init__(self, gain_pos_x0, gain_pos_y0,
                 gain_thrust0, gain_roll0, gain_pitch0, time_per_cycle) -> None:
        # Load the Mujoco model from XML file
        self.s_new = None
        self.m = mujoco.MjModel.from_xml_path('assets/swung_payload.xml')
        self.d = mujoco.MjData(self.m)
        self.DRONE_ID = 0
        self.s_last = [0, 0, 0, 0, 0, 0]
        self.s_curr = [0, 0, 0, 0, 0, 0]

        self.time_per_cycle = time_per_cycle
        self.gain_pos_x0 = gain_pos_x0
        self.gain_pos_y0 = gain_pos_y0
        self.gain_thrust0 = gain_thrust0
        self.gain_roll0 = gain_roll0
        self.gain_pitch0 = gain_pitch0
        self.last_errors0 = [0, 0, 0, 0, 0]
        self.i_errors0 = [0, 0, 0, 0, 0]

        self.timesteps = []
        self.x_curr_drone0 = []
        self.y_curr_drone0 = []
        self.z_curr_drone0 = []

        self.com_payload_x = []
        self.com_payload_y = []
        self.com_payload_z = []

        self.x_des = 0
        self.y_des = 0
        self.z_des = 5

        self.x_curr, self.y_curr, self.z_curr = self.current_pos()
        self.t = 0
        self.start = time.time()

    def reset(self):
        self.d.qpos = [0, 0, 2.25, 0, 0, 0, 0, 0, 0, 0.15, 0, 0, 0, 0]
        self.d.qvel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def roll_pitch_yaw(self):
        # Extract quaternion components from qpos
        if self.DRONE_ID == 0:
            q = self.d.qpos[3:7]
        else:
            q = self.d.qpos[17:21]

        # Calculate roll, pitch, yaw from quaternion
        pitch = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
        roll = math.asin(-2.0 * (q[1] * q[3] - q[0] * q[2]))
        yaw = - math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def PID(self, kp, kd, ki, error, last_error, i_err):
        # self.com_payload_y[0] = 0
        out = kp * error + kd * (error - last_error) + ki * i_err
        last_error = error
        i_err += error
        return out, last_error, i_err

    def current_pos(self):
        global x_curr, y_curr, z_curr
        if self.DRONE_ID == 0:
            x_curr = self.d.qpos[0]
            y_curr = self.d.qpos[1]
            z_curr = self.d.qpos[2]
        return x_curr, y_curr, z_curr

    def action_motor(self, control):
        if control[0] > 2.5:
            control[0] = 2.5
        if control[0] < 0:
            control[0] = 0
        if self.DRONE_ID == 0:
            self.d.ctrl[0] = control[0] - control[1] + control[2]
            self.d.ctrl[1] = control[0] + control[1] + control[2]
            self.d.ctrl[2] = control[0] + control[1] - control[2]
            self.d.ctrl[3] = control[0] - control[1] - control[2]

    def update_pos(self):
        # self.vel_x.append(float(self.d.qvel[0]))
        self.timesteps.append(time.time() - self.start)
        self.x_curr_drone0.append(self.d.qpos[0])
        self.y_curr_drone0.append(self.d.qpos[1])
        self.z_curr_drone0.append(float(self.d.qpos[2]))
        self.com_payload_x.append(self.d.qpos[7])
        self.com_payload_y.append(self.d.qpos[8])
        self.com_payload_z.append((self.d.qpos[9]))

    def controller(self):

        # last_errors = [x,y.thrust,roll,pitch]
        # i_errors = [x,y.thrust,roll,pitch]
        # gain_ = [kp,kd,ki]

        self.x_curr, self.y_curr, self.z_curr = self.current_pos()
        roll_curr, pitch_curr, _ = self.roll_pitch_yaw()
        ###### POSITION CONTROLLER #######
        ###### ROLL POSITION CONTROLLER #####

        if self.x_des != self.x_curr:
            error_x = self.x_des - self.x_curr
            out_x, self.last_errors0[0], self.i_errors0[0] = self.PID(self.gain_pos_x0[0], self.gain_pos_x0[1],
                                                                      self.gain_pos_x0[2], error_x,
                                                                      self.last_errors0[0], self.i_errors0[0])
            roll_des = out_x
        else:
            roll_des = 0

        ###### PITCH POSITION CONTROLLER #######

        # print("Z: ",z_curr)
        if self.y_des != self.y_curr:
            error_y = -self.y_des + self.y_curr
            out_y, self.last_errors0[1], self.i_errors0[1] = self.PID(self.gain_pos_y0[0], self.gain_pos_y0[1],
                                                                      self.gain_pos_y0[2], error_y,
                                                                      self.last_errors0[1], self.i_errors0[1])
            pitch_des = out_y
        else:
            pitch_des = 0

        # print(self.last_errors0[2])
        ########### THRUST CONTROLLER ############

        if float(self.z_curr) == self.z_des:
            force = 1.962
            self.last_errors0[2] = 0
        else:
            error_t = self.z_des - self.z_curr
            out_t, self.last_errors0[2], self.i_errors0[2] = self.PID(self.gain_thrust0[0], self.gain_thrust0[1],
                                                                      self.gain_thrust0[2], error_t,
                                                                      self.last_errors0[2], self.i_errors0[2])
            force = 1.962 + out_t
        thrust = force / 4

        # print(self.last_errors0[2])
        ########### ROLL CONTROLLER ############
        if roll_curr != roll_des:
            error_r = roll_des - roll_curr
            out_r, self.last_errors0[3], self.i_errors0[3] = self.PID(self.gain_roll0[0], self.gain_roll0[1],
                                                                      self.gain_roll0[2],
                                                                      error_r, self.last_errors0[3], self.i_errors0[3])
            roll = out_r
        else:
            roll = 0

        ########### PITCH CONTROLLER ############
        # print(self.last_errors0[4])
        if pitch_curr != pitch_des:
            error_p = pitch_des - pitch_curr
            out_p, self.last_errors0[4], self.i_errors0[4] = self.PID(self.gain_pitch0[0], self.gain_pitch0[1],
                                                                      self.gain_pitch0[2], error_p,
                                                                      self.last_errors0[4],
                                                                      self.i_errors0[4])
            pitch = out_p
        else:
            pitch = 0

        control = [thrust, roll, pitch]
        # print(control[0])
        return control

    def update_thrust0(self):
        self.gain_thrust0[0] = self.gain_thrust0[0] + 1
        self.gain_thrust0[1] = self.gain_thrust0[1] + 1
        self.gain_thrust0[2] = self.gain_thrust0[2] + 1

    @property
    def return_state(self):
        self.s_last = [self.s_curr[0], self.s_curr[1], self.s_curr[2],
                       self.s_curr[3], self.s_curr[4], self.s_curr[5]]

        self.s_curr = [self.d.qpos[0], self.d.qpos[1], self.d.qpos[2],
                       self.last_errors0[0], self.last_errors0[1], self.last_errors0[2]]

        return self.s_last, self.s_curr

    def get_action(self, lst):
        self.gain_thrust0 = lst
        return self.gain_thrust0

    def get_reward(self):
        # self.s_last, self.s_curr = self.return_state
        # reward = 100 - abs(self.s_last[2] - self.s_curr[2]) - 10 * abs(self.s_last[0] - self.s_curr[0]) - 10 * abs(
        # self.s_last[1] - self.s_curr[1])
        # reward = - abs(self.s_curr[2] - 5) + 5 - abs(self.s_curr[5] - self.s_last[5]) + 5
        reward = - abs(self.s_curr[2] - 5) + 5 # - abs(self.s_curr[0]) + 5
        # reward = 10 * self.s_curr[2] - self.s_curr[2] ** 2 + (self.s_last[5] - self.s_curr[5])
        return reward

    def run(self, action):
        # with mujoco.viewer.launch_passive(self.m, self.d) as viewer:

        # while time.time() - self.start < 50: #viewer.is_running() and
        step_start = time.time()
        self.t = round(time.time() - self.start, 4)

        self.m.opt.wind = (0, 0, 0)

        self.update_pos()
        # print(round(time.time() - (self.start), 4))
        # (action.self.s_curr(self.s_curr=drone.return_state[2]))

        if self.t % float(self.time_per_cycle) < 0.01:
            self.gain_thrust0 = [0, 0, 0]  # action
            # print(self.gain_thrust0)
            # s_old = s_new
            self.s, self.s_new = self.return_state
            # return self.s, self.s_new
            # print(drone.return_state[2])
            # drone.update_thrust0()
            # print(self.s, self.a, self.s_new)

        control_d0 = self.controller()
        mujoco.set_mjcb_control(self.action_motor(control_d0))

        mujoco.mj_step(self.m, self.d)

        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def logger_distance(self):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # Plot on the first subplot
        ax1.plot(self.timesteps, self.y_curr_drone0, label='y')
        ax1.plot(self.timesteps, self.z_curr_drone0, label='z')
        ax1.plot(self.timesteps, self.x_curr_drone0, label='x')
        ax1.set_xlabel('time')
        ax1.set_ylabel('distance')
        ax1.set_title("drone0")
        ax1.legend()
        # Plot on the second subplot

        # Adjust the spacing between subplots
        plt.tight_layout()
        # Display the plots
        plt.show()
