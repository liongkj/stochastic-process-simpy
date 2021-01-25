import simpy
import numpy as np
import pandas as pd
from numpy import random
import math
from helper import round_down

counter_total_service_times = 0
counter_total_idle_times = 0
column_names = ["Arrival Time","Current Q length","Q time","Service Start Time","Food prepare start","Exit system time","Food Prepare Duration","Total Wait Time(Queue+Food)","Service Time","Total Time in System"]

class Counter(object):
#     Counters to take order
    def __init__(self,env,num_counter):
        self.env = env
        self.counter = simpy.Resource(env,num_counter)
        self.counter_waiting = 0
        self.service_start = None
        
    def take_order(self,cus,env,service_start,parameters):
        print("%s is placing order at counter %.2f" %(cus,service_start))
        time_taken_to_place_order = max(random.exponential(scale = parameters['order_time_mu']),parameters['order_time_min'])
        yield self.env.timeout(time_taken_to_place_order)
        print("Order of %s sent to kitchen at %.2f" %(cus, env.now))
        # Record idle counter and add to total count
        
        
    def receive_order(self,cus,env,resource,service_start,parameters,data):
        global counter_total_idle_times
        global counter_total_service_times

        with resource.kitchen.request() as my_turn:
            yield my_turn
            yield env.process(resource.prepare_food(cus,env,data,parameters))
            service_end = env.now
            print("%s collected the food at %.2f" %(cus, service_end))
           
            counter_total_service_times += (service_end-service_start)
            

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
        yield env.process(queue.take_order(label,env,service_start,parameters))
        # waiting order at counter
        prepare_food_start = env.now
        data[label,4] = round_down(prepare_food_start)
        # counter is idle now
        yield env.process(queue.receive_order(label,env,kitchen,service_start,parameters,data))
        # prepare_food_end = round_down(env.now)
        # counter_total_wait_times += round_down(prepare_food_end - prepare_food_start)
        # receive food from counter
        exit_time = env.now
        data[label,5] = round_down(exit_time)

        # total wait time
        data[label,7] = round_down(data[label,5]+data[label,1])
        # total service time
        data[label,8] = round_down(exit_time-service_start)
        # total time in system
        data[label,9] = round_down(exit_time-arrive_time)
        yield env.timeout(0)


# Simlating possion process for customer arrival
def customer_arrivals(env,n_customer,res_counter,kitchen,parameters,result_fifo):
    """Create new *customer* until the sim time reaches 120. with poisson process"""
    for i in range(n_customer):
        yield env.timeout(random.poisson(1/parameters['lamb']))
        env.process(customer(env, i+1, res_counter, kitchen,parameters, result_fifo))


def startSimulation(n_customer,n_counter,n_kitchen,SIM_TIME,parameters):
    env = simpy.Environment()
    result_fifo = np.zeros((n_customer,len(column_names)))
    counter = Counter(env,n_counter)
    kitchen = Kitchen(env,n_kitchen)
    env.process(customer_arrivals(env,n_customer,counter,kitchen,parameters,result_fifo))
    env.run(until=SIM_TIME)

    labels = [*range(1,n_customer+1)]
    np_arr = np.array(result_fifo).reshape(n_customer,-1)
    df_fifo=pd.DataFrame(data = np_arr,index=labels,columns=column_names)
    df_fifo=df_fifo.drop(df_fifo[df_fifo.iloc[:,9]==0].index,axis=0) # remove unfinished customer
    total_wait_time = df_fifo.iloc[:,7].sum()
    total_service_time = df_fifo.iloc[:,8].sum()
    total_time_in_system = df_fifo.iloc[:,9].sum()
    total_counter_idle = df_fifo.iloc[:,6].sum()
    return df_fifo,total_wait_time,total_service_time,total_time_in_system,counter_total_service_times, total_counter_idle,SIM_TIME
