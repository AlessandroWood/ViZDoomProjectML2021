import json
import numpy as np

class Stats:
    stats_events = {}
    stats_file_path = "content/health/stats-health.json"
    stats_events_file_path = "content/health/stats-events-health.json"

    current_stats_event = {}

    def __init__(self):
       self.stats_events = {"reward": [], "medkit_pickup": [], "poison_pickup": [], "health": [], "distance": [], "velocity": [], "done": []}
       self.current_stats_event = {"reward": [], "medkit_pickup": [], "poison_pickup": [], "health": [], "distance": [], "velocity": [], "done": []}

    def record_events_stats_for_current_epoch(self, reward, medkit_pickup, poison_pickup, health, distance, velocity, done):
        self.current_stats_event["reward"].append(reward)
        self.current_stats_event["medkit_pickup"].append(medkit_pickup)
        self.current_stats_event["poison_pickup"].append(poison_pickup)
        self.current_stats_event["health"].append(health)
        self.current_stats_event["distance"].append(distance)
        self.current_stats_event["velocity"].append(velocity)
        self.current_stats_event["done"].append(done)


    def save_events_stats(self, epoch):

        reward = np.array(self.current_stats_event["reward"])
        medkit_pickup = np.array(self.current_stats_event["medkit_pickup"])
        poison_pickup = np.array(self.current_stats_event["poison_pickup"])
        health = np.array(self.current_stats_event["health"])
        distance = np.array(self.current_stats_event["distance"])
        velocity = np.array(self.current_stats_event["velocity"])
        done = np.array(self.current_stats_event["done"])

        self.stats_events["reward"].append({"min": reward.min().item(), "max": reward.max().item(), "mean": reward.mean().item(),"std": reward.std().item()})
        self.stats_events["medkit_pickup"].append({"min": medkit_pickup.min().item(), "max": medkit_pickup.max().item(), "mean": medkit_pickup.mean().item(), "std": medkit_pickup.std().item()})
        self.stats_events["poison_pickup"].append({"min": poison_pickup.min().item(), "max": poison_pickup.max().item(), "mean": poison_pickup.mean().item(), "std": poison_pickup.std().item()})
        self.stats_events["health"].append({"min": health.min().item(), "max": health.max().item(), "mean": health.mean().item(), "std": health.std().item()})
        self.stats_events["distance"].append({"min": distance.min().item(), "max": distance.max().item(), "mean": distance.mean().item(), "std": distance.std().item()})
        self.stats_events["velocity"].append({"min": velocity.min().item(), "max": velocity.max().item(), "mean": velocity.mean().item(), "std": velocity.std().item()})
        self.stats_events["done"].append({"min": done.min().item(), "max": done.max().item(), "mean": done.mean().item(), "std": done.std().item()})
        self.stats_events["epoch"] = epoch

        with open(self.stats_events_file_path, 'w') as f:
            json.dump(self.stats_events, f, indent=4)

        self.current_stats_event = {"reward": [], "medkit_pickup": [], "poison_pickup": [], "health": [], "distance": [], "velocity": [], "done": [], "epoch": 0}



