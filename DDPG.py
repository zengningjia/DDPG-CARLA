from socket import SO_SNDTIMEO
import time
import numpy as np
from keras import layers
import tensorflow as tf
import keras
import os
import argparse
import keras.backend.tensorflow_backend as backend
import random
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.prev_measurements = None

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    #action 需要是一个数组（或者字典？）
    def step(self, action):
        steer = action['steer']
        acc = action['acc']
        brake = action['brake']
        self.vehicle.apply_control(carla.VehicleControl(throttle=acc))
        self.vehicle.apply_control(carla.VehicleControl(steer=steer))
        self.vehicle.apply_control(carla.VehicleControl(brake=brake))
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        return self.front_camera, reward, done, None

MAX_EPISODES = 1
MAX_EP_STEP = 1
TEST_PER_EPISODES = 10

MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
VAR = 3
class DDPGAgent:
    def __init__(self, s_dim, a_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim*2+a_dim+1), dtype=np.float32)
        self.pointer = 0

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.R = layers.Input([None,1], tf.float32, 'r')

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            inputs = layers.Input(input_state_shape, name='A_input')
            x = layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = layers.Lambda(lambda x : np.array(a_bound) * x)(x)
            return keras.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name = ''):
            s = layers.Input(input_state_shape, name='C_s_input')
            a = layers.Input(input_action_shape, name='C_a_input')
            x = layers.Concat(1)([s,a])
            x = layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return keras.models.Model(inputs=[s,a], outputs=x, name='Critic'+name)
        
        # 生成critic和actor网络
        self.critic = get_critic([None, s_dim], [None,a_dim])
        self.actor = get_actor([None, s_dim])
        #这两行是什么意思？
        self.critic.train()
        self.actor.train()

        #生成target网络
        def copy_para(from_model, to_model):
            for i,j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.critic_target = get_critic([None, s_dim], [None, a_dim], name = '_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        # 设置优化器
        self.critic_opt = tf.optimizers.Adam(LR_C)
        self.actor_opt = tf.optimizers.Adam(LR_A)

        # 设置target网络更新方式，软更新,1-TAU表示旧元素的比例
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU) 

        def ema_update(self):
            paras = self.actor.trainable_weights + self.critic.trainable_weights
            self.ema.apply(paras)
            for i,j in zip(paras,self.actor_target.trainable_weights+self.critic_target.trainable_weights):
                j.assign(self.ema.average(i))

        # actor choose an action
        def store_transition(self, s, a, r, s_):
            s = s.astype(np.float32)
            s_ = s_.astype(np.float32)
            transition = np.hstack((s, a, [r], s_)) # stack in horizon way
            index = self.pointer % MEMORY_CAPACITY # pointer record how many exp totally
            self.memory[index, :] = transition
            self.pointer += 1
        
        def learn(self):
            # random sample 
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
            bt = self.memory[indices, :] # batch
            bs = bt[:, :self.s_dim] # state
            ba = bt[:, self.s_dim:self.s_dim+self.a_dim] # action
            br = bt[:, -self.s_dim-1:-self.s_dim] # reward
            bs_ = bt[:, -self.s_dim:] # state_

            # update Critic
            with tf.GradientTape() as tape:
                a_ = self.actor_target(bs_)
                q_ = self.critic_target([bs_, a_])
                y = br + GAMMA * q_
                q = self.critic([bs, ba])
                td_error = tf.losses.mean_squared_error(y,q)
            c_grads = tape.gradient(td_error, self.critic.trainable_weights)
            self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))
            
            # update Actor
            with tf.GradientTape() as tape:
                a = self.actor(bs)
                q = self.critic([bs, a])
                a_loss = -tf.reduce_mean(q) # calculate the mean value of all the q
            a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
            self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

            self.ema_update()

            # 保存、加载模型
            def save_ckpt(self):
                if not os.path.exists('model'):
                    os.makedirs('model')
                keras.models.save_model(self.actor, 'model/ddpg_actor.hdf5')
                keras.models.save_model(self.actor_target, 'model/ddpg_actor_target.hdf5')
                keras.models.save_model(self.critic, 'model/ddpg_critic.hdf5')
                keras.models.save_model(self.critic_target, 'model/ddpg_critic_target.hdf5')
            def load_ckpt(self):
                actor = keras.models.load_model('model/ddpg_actor.hdf5')
                actor_target = keras.models.load_model('model/ddpg_actor_target.hdf5')
                critic = keras.models.load_model('model/ddpg_critic.hdf5')
                critic_target = keras.models.load_model('model/ddpg_critic_target.hdf5')
                return actor, actor_target, critic, critic_target

if __name__ == '__main__':

    # 添加命令行指令
    parser = argparse.ArgumentParser(description='确定训练还是测试')
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='test', action='store_false')
    args = parser.parse_args()

    FPS = 30
    ep_reward = [-200]

    # 保证实验可复现
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when training multiple agents
    def gpu_memory_config(mode, ratio = 0.7):
        config = tf.ConfigProto()
        if (mode == 'adaption'):
            config.gpu_options.allow_growth = True
        if (mode == 'ratio'):
            config.gpu_options.per_process_gpu_memory_fraction = ratio
        return config
    config = gpu_memory_config('adaption')
    session = tf.Session(config = config)
    backend.set_session(session)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))         

    for i in range(MAX_EPISODES):
        t1