import simpy
import numpy as np
import pandas as pd
from numpy import random
import math
from helper import round_down

counter_total_service_times = 0
column_names = ["Arrival Time","Current Q Length","Q time","Service Start Time","Food prepare start","Exit system time","Food Prepare Duration","Total Wait Time(Queue+Food)","Service Time","Total Time in System"]

class Counter(object):
#     Counters to take order
    def __init__(self,env,num_counter):
        self.env = env
        self.counter = simpy.Resource(env,num_counter)
        self.counter_waiting = 0
        
    def take_order(self,cus,env,service_start,parameters,data):
        global counter_total_service_times

        print("%s is placing order at counter %.2f" %(cus,service_start))
        time_taken_to_place_order = max(random.exponential(scale = parameters['order_time_mu']),parameters['order_time_min'])
        yield self.env.timeout(time_taken_to_place_order)
        service_end = env.now
        print("Order of %s sent to kitchen at %.2f" %(cus, service_end))
        data[cus,8] = round_down(service_end-service_start)
        if(cus<1000):
            counter_total_service_times += service_end-service_start

    def receive_order(self,cus,env,kitchen,parameters,data):
        
        with kitchen.kitchen.request() as my_turn:
            yield my_turn
            yield env.process(kitchen.prepare_food(cus,env,data,parameters))
            food_end = env.now
            print("%s collected the food at %.2f" %(cus, food_end))
            # Record idle counter and add to total count
           
class Kitchen(object):
    # Kitchen to prepare food
    def __init__(self,env,num_kitchen):
        self.env = env
        self.kitchen = simpy.Resource(env,num_kitchen)
    
    def prepare_food(self,cus,env,data,parameters):
        print("Kitchen is preparing food for %s at %.2f" %(cus, env.now))
        food_prepare_time = max(parameters['food_prepare_min'],random.exponential(scale = parameters['food_prepare_mu']))
        data[cus,6] = round_down(food_prepare_time)
        yield self.env.timeout(food_prepare_time)
        print("Cooked food for %s at %.2f" %(cus, env.now))

def customer(env, label, queue, kitchen,parameters, data):
#     the customer process arrive at the restaurant and request counter to take order
    label = label-1
    arrive_time = env.now
    print("%s entering the queue at %.2f"%(label,arrive_time))
#     data[label,0]=label
    data[label,0]= round_down(arrive_time)
    data[label,1] = len(queue.counter.queue)
    with queue.counter.request() as my_turn:
        yield my_turn
        service_start = env.now
        data[label,3] = round_down(service_start)
        queue_time = service_start - arrive_time
        data[label,2]= round_down(queue_time)
        # placing order at counter
        yield env.process(queue.take_order(label,env,service_start,parameters,data))
        # waiting order at counter
        prepare_food_start = env.now
        data[label,4] = round_down(prepare_food_start)
        yield env.timeout(0)
    yield env.process(queue.receive_order(label,env,kitchen,parameters,data))
    # prepare_food_end = round_down(env.now)
    # counter_total_wait_times += round_down(prepare_food_end - prepare_food_start)
    # receive food from counter
    exit_time = env.now
    data[label,5] = round_down(exit_time)

    # total wait time
    data[label,7] = round_down(data[label,6]+data[label,2])
    # total time in system
    data[label,9] = round_down(exit_time-arrive_time)

# Simlating possion process for customer arrival
def customer_arrivals(env,n_customer,res_counter,kitchen,parameters,result_ticket):
    """Create new *customer* until the sim time reaches 120. with poisson process"""
    for i in range(n_customer):
        yield env.timeout(random.poisson(1/parameters['lamb']))
        env.process(customer(env, i+1, res_counter, kitchen,parameters, result_ticket))


def startSimulation(n_customer,n_counter,n_kitchen,SIM_TIME,parameters):
    env = simpy.Environment()
    result_ticket = np.zeros((n_customer,len(column_names)))
    counter = Counter(env,n_counter)
    kitchen = Kitchen(env,n_kitchen)
    env.process(customer_arrivals(env,n_customer,counter,kitchen,parameters,result_ticket))
    env.run(until=SIM_TIME,)
#     env.run(until=proc)

    labels = [*range(1,n_customer+1)]
    np_arr = np.array(result_ticket).reshape(n_customer,-1)
    df_ticket=pd.DataFrame(data = np_arr,index=labels,columns=column_names)
    df_ticket=df_ticket.drop(df_ticket[df_ticket.iloc[:,-1]==0].index,axis=0) # remove unfinished customer
    df_ticket=df_ticket.iloc[:1000]
    total_wait_time = df_ticket.iloc[:,7].sum()
    total_service_time = df_ticket.iloc[:,8].sum()
    total_time_in_system = df_ticket.iloc[:,9].sum()
    counter_total_idle_times = (n_counter*SIM_TIME - total_service_time)
    sim_time = df_ticket.iloc[-1,5]
    return df_ticket,total_wait_time,total_service_time, total_time_in_system,counter_total_service_times, counter_total_idle_times,sim_time
