#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Luca Gregori, Alessandro Wood
# basato sul codice di E. Culurciello, L. Mueller, Z. Boztoprak
# Marzo - Luglio 2021

from __future__ import print_function
import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
import random
import itertools as it
import skimage.transform
from vizdoom import Mode
from time import sleep, time
from collections import deque
from tqdm import trange
import json
import math
import torch.nn.functional as F
from event_buffer import EventBuffer
from event_stats import Stats

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 80
learning_steps_per_epoch = 2000
target_net_update_steps = 10000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (120, 160)
episodes_to_watch = 100

model_savefile = "content/health-gathering-supreme/model-doom-health.pth"
save_model = True
load_model = False
skip_learning = False

description = "health gathering supreme con ROE"
dueling = True

info = {"description": description,
        "resolution": resolution,
        "frame_repeat": frame_repeat,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "train_epochs": train_epochs,
        "learning_steps_per_epoch": learning_steps_per_epoch,
        "replay_memory_size": replay_memory_size,
        "batch_size": batch_size,
        "test_episodes_per_epoch": test_episodes_per_epoch,
        "net_descritpion": "",
        "net_parameters:": 0,
        "dueling Double DQN": dueling}

# Stats
save_stats = True
e_stats = Stats()
e_stats.__init__()
stats = {"loss": [], "train_scores": [], "test_scores": [], "time": [], "epochs": 0, "info": info}
stats_events = [{"intrinsic_reward": [], "medkit_pickup": [], "poison_pickup" : [], "health": [], "distance": [], "velocity": [], "step": []}, None]
stats_file_path = "content/health-gs/stats-health-gs.json"
stats_events_file_path = "content/health-gathering-supreme/stats-events-health-gs.json"

# Configuration file path
config_file_path = "content/scenarios/health_gathering_supreme.cfg"

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')


def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())
    if save_stats:
        stats["test_scores"].append(
            {"min": test_scores.min(), "max": test_scores.max(), "mean": test_scores.mean(), "std": test_scores.std()})


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()
    for epoch in range(num_epochs):
        game.new_episode()
        last_positions = deque(maxlen=25)
        train_scores = []
        train_error = torch.zeros(steps_per_epoch).to(DEVICE)
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))
        last_total_medikit_reward = 0
        last_total_poison_reward = 0
        list_of_events = []
        for step in trange(steps_per_epoch, leave=False):
            current_events = []
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            medikit_reward = game.get_game_variable(vzd.GameVariable.USER1)
            poison_reward = - game.get_game_variable(vzd.GameVariable.USER2)
            health = game.get_game_variable(vzd.GameVariable.HEALTH)
            x = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.POSITION_X))
            y = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.POSITION_Y))
            vel_x = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.VELOCITY_X))
            vel_y = vzd.doom_fixed_to_double(game.get_game_variable(vzd.GameVariable.VELOCITY_Y))
            last_positions.append((x, y))
            last_x, last_y = last_positions[0]
            dist = math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2) * 100
            vel = math.sqrt((vel_x) ** 2 + (vel_y) ** 2) * 10000

            medikit_reward = vzd.doom_fixed_to_double(medikit_reward)
            poison_reward = vzd.doom_fixed_to_double(poison_reward)
            medikit_reward = medikit_reward - last_total_medikit_reward
            poison_reward = poison_reward - last_total_poison_reward
            medikit_pickup = medikit_reward / 100
            poison_pickup = (poison_reward / 100)

            last_total_medikit_reward += medikit_reward
            last_total_poison_reward += poison_reward

            current_events.append(2.5 * medikit_pickup)
            current_events.append(-1.25 * poison_pickup)
            current_events.append(health/100)
            current_events.append(dist * 0.5)
            current_events.append(vel * 1.75)

            reward = agent.event_buffer.intrinsic_reward(current_events)
            current_events[0] = medikit_pickup
            current_events[1] = poison_pickup
            current_events[3] = dist
            current_events[4] = vel
            list_of_events.append(current_events)

            done = game.is_episode_finished()

            e_stats.record_events_stats_for_current_epoch(reward, medikit_pickup, poison_pickup, health, dist, vel, done)

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 120, 160)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                td_error = agent.train()
                train_error[global_step] = td_error

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()
                last_positions.clear()
                last_total_medikit_reward = 0
                last_total_poison_reward = 0
                events_mean = np.mean(np.array(list_of_events), axis=0)
                agent.event_buffer.record_events(events_mean)
                list_of_events = []

            if (((global_step + (epoch * learning_steps_per_epoch)) % target_net_update_steps) == 0):
                agent.update_target_net()

            global_step += 1

        # agent.update_target_net()
        train_scores = torch.tensor(train_scores).to(DEVICE)

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)

        if save_stats:
            print("Saving stats to:", stats_file_path)
            stats["train_scores"].append(
                {"min": train_scores.min().item(), "max": train_scores.max().item(), "mean": train_scores.mean().item(),
                 "std": train_scores.std().item()})
            stats["loss"].append(
                {"min": train_error.min().item(), "max": train_error.max().item(), "mean": train_error.mean().item(),
                 "std": train_error.std().item()})
            stats["epochs"] += 1
            with open(stats_file_path, 'w') as f:
                json.dump(stats, f, indent=4)

            print("Saving stats events to:", e_stats.stats_events_file_path)
            e_stats.save_events_stats(epoch + 1)

        t = time() - start_time
        stats["time"].append(t / 60)
        print("Total elapsed time: %.2f minutes" % (t / 60.0))

    game.close()
    return agent, game


class DuelQNet(nn.Module):
    """
    This is a Dueling DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """
    def __init__(self, available_actions_count):
        super(DuelQNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.state_fc = nn.Sequential(
            nn.Linear(12 * 34 * 64, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )
        self.advantage_fc = nn.Sequential(
            nn.Linear(12 * 34 * 64, 512),
            nn.ELU(),
            nn.Linear(512, available_actions_count)
        )

    #@torch.jit.script_method
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 24 * 34 * 64)
        x1 = x[:, :12 * 34 * 64]  # input for the net to calculate the state value
        x2 = x[:, 12 * 34 * 64:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_probs(self, inputs):
        value, x = self(inputs)
        x = self.dist(x)
        probs = F.softmax(x)
        return probs

class DQNAgent:
    def __init__(self, action_size, memory_size, batch_size, discount_factor,
                 lr, load_model, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.0, n = 5, event_buffer_capacity=100):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.event_buffer = EventBuffer(n, event_buffer_capacity)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min
        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        # self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)
        self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def freeze_layer_controller(self, epoch, epochRatio):
        if epoch < epochRatio:
            self.q_net.conv1.requires_grad_(False)
            self.q_net.conv2.requires_grad_(False)
            self.q_net.state_fc.requires_grad_(True)
            self.q_net.advantage_fc.requires_grad_(True)
        elif epoch < epochRatio * 2:
            self.q_net.conv2.requires_grad_(True)
        else:
            self.q_net.conv1.requires_grad_(True)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)
        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = torch.arange(self.batch_size)  # used for indexing the batch
        q_targets = torch.from_numpy(rewards).float().to(DEVICE)

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, torch.argmax(self.q_net(next_states), 1)
            next_state_values = self.target_net(next_states)[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets[not_dones] += self.discount * next_state_values

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return td_error


if __name__ == '__main__':
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model)

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                          steps_per_epoch=learning_steps_per_epoch)

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    stats["info"]["net_descritpion"] = str(summary(agent.q_net, (1, 120, 160)))
    stats["info"]["net_parameters"] = agent.q_net.count_parameters()
    if save_stats:
        with open(stats_file_path, 'w') as f:
            json.dump(stats, f, indent=4)
    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
