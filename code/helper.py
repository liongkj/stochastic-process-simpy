import math

#helper function
def round_down(n, decimals=1):
    multiplier = 10 ** decimals
    return abs(math.floor(n*multiplier + 0.5) / multiplier)

def print_sim_results(results,n_kitchen,n_counter):
    df_fifo,total_wait_time,total_service_time,total_time_in_system,counter_total_service_times, counter_total_idle_times, sim_time =results
    i = df_fifo.shape[0]
    print()
    print("%d Customers served"%i)
    print("Total Simulation Time=> %.2f Minutes" % sim_time)
    print("Total Idle Time for %d Counters=> %.2f Minutes" % (n_counter,counter_total_idle_times))
    print("Total Service Time for %d Counters=> %.2f Minutes" % (n_counter, counter_total_service_times))
    
    print("Average Queue Length=> %d " % df_fifo.iloc[:,1].mean())
    print("Max Queue Length=> %d " % df_fifo.iloc[:,1].max())
    print()
    print("Average Waiting Time => %.2f Minutes" % (total_wait_time / i))
    print("Average Service Time => %.2f Minutes" % (total_service_time / i))
    print("Average Time Spent In System => %.2f Minutes" % (total_time_in_system / i))
