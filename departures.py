import random 
from random import seed
import numpy as np
import simpy 

randome_seed = 42



def train(env, name, stationplatform):
    # 1. Arrival
    print(f'{name} arrives at the station at {env.now:.2f}')
    
    with stationplatform.request() as request:
        yield request
        
        # 2. Boarding Process for 5 Groups
        print(f'  [Boarding] {name} starts boarding groups...')
        
        total_boarding_time = 0
        for group_id in range(1, 6):
            # Simulate number of passengers in this group
            passengers = random.randint(10, 30)
            # Time taken per passenger (e.g., 0.05 to 0.1 minutes)
            group_time = passengers * random.uniform(0.05, 0.1)
            
            yield env.timeout(group_time)
            total_boarding_time += group_time
            print(f'    Group {group_id} finished boarding at {env.now:.2f}')

        # 3. Departure
        # Add a small buffer for doors closing
        yield env.timeout(1) 
        print(f'{name} is departing at {env.now:.2f} (Total dwell: {total_boarding_time + 1:.2f} min)')
        
def setup(env, num_stationplatforms):
    platform = simpy.Resource(env, capacity=num_stationplatforms)
    
    train_id = 0
    while True: 
        yield env.timeout(random.uniform(15, 25)) # Headway between trains
        train_id += 1
        env.process(train(env, f'Train {train_id}', platform))

# Run the simulation
env = simpy.Environment()
env.process(setup(env, 1)) 
env.run(until=100)