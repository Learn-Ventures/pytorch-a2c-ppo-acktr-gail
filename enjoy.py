import argparse, imageio
import os, shutil, sys, torch, cv2
import numpy as np

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
sys.path.append('a2c_ppo_acktr')
sys.path.append('/content/pybullet-gym')
import pybulletgym  # register PyBullet enviroments with open ai gym

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log-interval',
    type=int, default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--video-dir',
    default='./episode_videos/',
    help='directory to save videos (default: ./episode_videos/)')
parser.add_argument('--non-det',
    action='store_true',default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument('--nr_eps',
    type=int, default=100,
    help='The number of evaluation episodes to run')
parser.add_argument('--nr_videos',
    type=int, default=10,
    help='The number of evaluation episodes to save as video')

args = parser.parse_args()
args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1, None, None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    allow_early_resets=False)

render_func = get_render_func(env)
render_and_dump = True
# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)
obs = env.reset()

if render_func is not None:
    render_func('rgb_array')

def round_down(num, divisor):
    return num - (num%divisor)

total_reward, episode_reward, nr_episodes = 0, 0, 0
try: shutil.rmtree(args.video_dir)
except: pass

print("Running agent in evaluation mode...\n\n")
os.makedirs(args.video_dir, exist_ok = True)
print("\nSaving videos to %s\n\n" %args.video_dir)
writer = imageio.get_writer('%s/episode_%04d.mp4' %(args.video_dir, nr_episodes), fps=24)

import warnings
warnings.simplefilter('ignore')

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action) #hereX
    episode_reward += reward
    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1 and 0:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if (render_func is not None) and render_and_dump:
        frame = render_func('rgb_array')
        resized_frame = cv2.resize(frame, (round_down(frame.shape[0], 16), round_down(frame.shape[1], 16)))
        writer.append_data(resized_frame)

    if done:
      total_reward += episode_reward
      nr_episodes += 1
      print("------\tEpisode %d finished (with reward %d), avg-reward / episode: %.2f" %(nr_episodes, episode_reward, total_reward/nr_episodes))
      episode_reward = 0

      if render_and_dump:
        writer.close()
        writer = imageio.get_writer('%s/episode_%04d.mp4' %(args.video_dir, nr_episodes), fps=20)

      if nr_episodes > args.nr_videos:
        render_and_dump = False

      if nr_episodes > args.nr_eps:
        print("Agent evaluated for %d episodes, breaking!" %args.nr_eps)
        break
