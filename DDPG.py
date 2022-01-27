from socket import SO_SNDTIMEO
import time
from xml.dom.pulldom import PROCESSING_INSTRUCTION
import numpy as np
from keras import layers
import tensorflow as tf
import keras
import os
import argparse
import keras.backend.tensorflow_backend as backend
import random
import cv2
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
class action_space:
    def __init__(self, dim, high, low, seed):
        self.shape = (dim,)
        self.high = np.array(high)
        self.low = np.array(low)
        self.seed = seed
        # action space 是一维的，该维度有dim个数，每个数有自己的上下限
        # 为了给每一个数设计上下限，对应的high和low也是一维的，这个维度上有dim个数
        assert(dim == len(high) == len(low))
        np.random.seed(self.seed)
    
    def sample(self): # 从一个均匀分布中随机采样
        return np.random.uniform(self.low + (0,0.8), self.high)

class observation_sapce:
    def __init__(self, dim, high=None, low=None, seed=None):
        self.shape = (dim,)
        self.high = high
        self.low = low
        self.seed = seed

# class ImitationLearning:
#     def __init__(self, image):
#         self.image = image

#     def compute_feature(self):
#         # 处理图像

#         return vector




class Env:
    SHOW_CAM = False
    im_width = 320
    im_height = 240
    front_camera = None

    def __init__(self, MONITO_DIR, SEED, FPS, sess, action_lambda=0.5):
        
        # carla环境对象
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        
        # 保存训练图片
        self.MONITOR_DIR = MONITO_DIR # 训练图像文件夹
        self.image_dir_ = None # 训练图像文件夹
        self.image_i_ = 0 # 第几张训练图像

        # self.Image_agent = ImitationLearning() # 这个是用来处理图像的

        # 动作空间有两个动作，其上下限分别为(1,1)与(-1,-1)
        self.action_space = action_space(2, (1.0, 1.0), (-1.0, -1.0), SEED)
        # 状态空间
        self.observation_space = observation_sapce(512 + 3 + 2)    

        self.render_ = False # 是否保存训练时的图像
        self.action_lambda = action_lambda # 新动作所占比例

        self.FPS = FPS
        self.reward_time = 0

        self.prev_measurements = None # 上一个状态
        self.prev_action = {'steer':0.0, 'acc':0.0, 'brake':0.0} # 上一次采取的动作
    
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action): # step 返回的不止是一个图像，还有现在的速度,转向,油门
                            # 传入的action应该是一个字典？
    
        steer = action['steer'] * self.action_lambda + (1-self.action_lambda) * self.prev_action['steer']
        brake = action['brake'] * self.action_lambda + (1-self.action_lambda) * self.prev_action['brake']
        # prev_measurement是上一次的状态(其实就是速度),如果速度过大,要强制调整这次的油门
        acc = action['acc'] * self.action_lambda + (1-self.action_lambda) * self.prev_action['acc']
        if self.prev_measurements is not None and self.prev_measurements>=8:
            if acc > 0.5:
                acc = 0.5
            else:
                pass
        self.vehicle.apply_control(carla.VehicleControl(throttle=acc))
        self.vehicle.apply_control(carla.VehicleControl(steer=steer))
        self.vehicle.apply_control(carla.VehicleControl(brake=brake))
        

        # 如果需要，就训练保存图片
        if self.render_:
            im = self.front_camera
            if not os.path.isdir(os.path.join(self.MONITOR_DIR, self.image_dir_)):
                os.makedirs(os.path.join(self.MONITOR_DIR, self.image_dir_))
            im.save(os.path.join(self.MONITOR_DIR, self.image_dir_, str(self.image_i_) + '.jpg'))

        # 这里的measurements变量不应只包含速度，还需包含车路关系，碰撞关系，与目的地关系
        measurements = self.vehicle.get_velocity()

        speed = int(3.6 * math.sqrt(measurements.x**2 + measurements.y**2 + measurements.z**2)) # 转化成km/h，注意get_velocity得到的速度包含三个方向x,y,z

        reward, done = self.reward(measurements, self.prev_measurements, action,
        self.prev_action)

        self.prev_measurements = measurements
        self.prev_action['steer'] = steer
        self.prev_action['brake'] = brake
        self.prev_action['acc'] = acc
        info = 0

        return np.concatenate(self.front_camera, (steer, acc-brake, speed)), reward, done, info


    def reset(self):
        print('start to reset env')
        self.image_i_ = 0
        self.image_dir = None
        self.render_ = False
        self.reward_time = 0
        self.prev_measurements = None
        self.prev_action = {'steer':0.0, 'acc':0.0, 'brake':0.0}

        # 初始化carla环境,添加车辆,传感器等
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
        for i in range(10):
            action = {'steer': 0.0, 'acc': 0.0, 'brake': 0.0}
            observation, _, _, _ = self.step(action)
            time.sleep(0.1)   

        print('reset finished')
        return observation
    
    def render(self, image_dir):
        self.render_ = True
        self.image_dir_ = image_dir
    
    def reward(self, measurements, prev_measurements, action, prev_action):

        # reward与车辆状态(车速，转向，是否碰撞，车路关系，与目的地的距离)，动作有关
        done = False
        reward = 0.0

        '''考虑碰撞'''
        if len(self.collision_hist) != 0:
            done = True
            reward = -30

        '''考虑道路因素'''
        # 待添加

        '''考虑车速'''
        speed = measurements
        if speed <= 6:
            reward = reward + speed**2 / 6.0
        elif speed <= 8:
            reward = reward + 3 * (8 - speed)
        else:
            reward = reward - 2 * (speed - 8) ** 2
        
        '''考虑转向因素'''
        actual_steer = action['steer'] * self.action_lambda + (1 - self.action_lambda) * prev_action['steer']
        reward = reward - 4 * np.abs(actual_steer) * np.abs(actual_steer) * speed

        # 如果是第一步，就不考虑前一步的影响
        if prev_measurements is None:
            return reward, done
        
        '''考虑与目的地关系（measurement未包含此信息，待添加）'''
        # x, y = measurements.x, measurements.y
        # pre_x, pre_y = prev_measurements.x, prev_measurements.y
        # distance = np.sqtr((x-pre_x)**2 + (y-pre_y)**2)
        # if distance < 1/self.FPS or speed < 0:
        #     reward = reward - 2 
        #     if speed < 0:
        #         self.reward_time += 5
        #     self.reward += 1
        # else:
        #     self.reward_time = 0
        # if self.reward_time >= 20:
        #     done = True
        #     self.reward_time = 0

        return reward, done



class DDPGAgent:
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
    def __init__(self, s_dim, a_dim, a_bound):
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim*2+a_dim+1), dtype=np.float32)
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

        def copy_para(from_model, to_model):
            for i,j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)
        
        # 生成critic和actor网络
        self.critic = get_critic([None, s_dim], [None,a_dim])
        self.actor = get_actor([None, s_dim])
        self.critic.train() # ？
        self.actor.train()

        # 生成target网络
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name = '_target')
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        copy_para(self.actor, self.actor_target)
        self.critic_target.eval() # ？
        self.actor_target.eval()

        # 设置优化器
        self.critic_opt = tf.optimizers.Adam(self.LR_C)
        self.actor_opt = tf.optimizers.Adam(self.LR_A)

        # 设置target网络更新方式，软更新,1-TAU表示旧元素的比例
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU) 

    def ema_update(self):
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i,j in zip(paras,self.actor_target.trainable_weights+self.critic_target.trainable_weights):
            j.assign(self.ema.average(i))

    # actor choose an action
    def store_transition(self, s, a, r, s_):
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_)) # 横向堆叠
        index = self.pointer % self.MEMORY_CAPACITY # pointer 记录一共有多少条经验
        self.memory[index, :] = transition
        self.pointer += 1
        
    def learn(self):
        # random sample 
        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :] # batch
        bs = bt[:, :self.s_dim] # state
        ba = bt[:, self.s_dim:self.s_dim+self.a_dim] # action
        br = bt[:, -self.s_dim-1:-self.s_dim] # reward
        bs_ = bt[:, -self.s_dim:] # state_

        # 更新critic
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + self.GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y,q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))
        
        # 更新actor
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

    # 实例化
    a_dim = 
    ddpg = DDPG(a_dim, s_dim, a_bound)

    for i in range(MAX_EPISODES):
        t1