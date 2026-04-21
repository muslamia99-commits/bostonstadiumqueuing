import simpy
import random

def train(env, name, station):
    print(f'{name} arrives at the station at time {env.now}')
    yield env.timeout(random.randint(1, 3))  # Simulate time taken for passengers to board
    print(f'{name} departs from the station at time {env.now}')