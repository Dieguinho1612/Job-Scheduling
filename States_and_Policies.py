#!/usr/bin/env python
# coding: utf-8

# # States and Policies

# ## Description of Notebook
# In this Notebook we will create the states for the Job Scheduling Problem in the deterministic case.<br>
# We will define the class of states and give a function to compute every state.<br>
# To be able to feed them into a Neural Network later on, we will add a compatible representation of their information as data for the input.<br>
# Since we approach this problem from the perspective of Deep Q-Learning, their can be taken an action in every non-final state, leading to one of the successor states. Therefore, every state needs a Q-value for every corresponding action that is possible. Either these Q-Values or the corresponding one-hot-vector indicating the optimal action will then become the target-vector.<br>
# Finally, we will define a composite function executing all these steps.<br>
# From here on, it is easy to compute Policies, including the Optimal Policy. A function herefore for will be given as well.

# ## Code

# In[1]:


#import dependencies
import operator
import random
import numpy as np
import copy
import time
import pickle
#import import_ipynb
from Jobs_and_Machines import *


# ### Class of States
# The idea is that the Machines will be processing the Jobs. Whenever a Job finishes, a Machine gets free and therefore an action has to be taken. This situation will be given in the form of a State.<br>
# Possible actions in this state are to assign a new Job to a Machine or to shut the free Machine down, which means it cannot be used anymore from here on.<br>
# If all Jobs have been assigned and/or processed already, the state is final and no decision has to be taken.<br>
# To save computing time, for the case that more than one Machine is free at some point we will always only use the free Machine with the lowest index. After having taken an action on it, it is not free anymore and we move to the next state, which will happen to have the same time, to take care of the next free Machine.
# <br>This approach could lead to splitting the Reinforcement Learning Problem into several ones, having one agent for every Machine. However, when passing the information about the state to the Neural Network, we will also pass the information of which Machines are free in this state to counteract the mentioned tendency. 
# 

# A State consists of these environmental information:
# 
#    1. The Jobs and Machines of the Job Scheduling Problem it belongs to
#    2. Time
#    3. Remaining Jobs
#    4. Machines that are still on duty
#    5. Free Machines
#    6. Remaining processing time of occupated Machines
#    7. Which Machine is working on which Job
#     
# The State also needs information about how it is related to other States:
#    1. ID of State
#    2. Predecessor State
#    3. Action that led to this state
#    4. Transition Costs from predecessor to current State
#    5. The machine that will be used
#    6. A dictionary mapping every action to the corresponding succesor states as well as a list of all successor States
#    7. Optimal Future Costs for every action (Q-Values)
#    
# Finally, it needs its information stored as data that the Neural Network can process:
#    1. Input
#    2. Target

# In[2]:


class states:
    def __init__(self, time, jobs_remaining, machines_on_duty,
                 free_machines, machine_runtimes, machine_occupations, predecessor):
        
        #job scheduling problem environment
        self.jobs = list_jobs
        self.machines = list_machines
        
        #time
        self.time = time
        
        #information about the jobs
        self.jobs_remaining = jobs_remaining #"one" if job is still remaining, "zero" if it was assigned already
        
        #information about the machines
        self.machines_on_duty = machines_on_duty #"one" if machine is still on duty, "zero" if it was shut down already
        self.free_machines = free_machines #"one" if machine is free, "zero" if it is occupied or shut down
        self.machine_runtimes = machine_runtimes #remaining runtime of every machine
        self.machine_occupations = machine_occupations #which machine is working on which job
        
        #predecessor and successors
        self.ID = None #ID of the state, will be given after all states were created
        self.predecessor = predecessor #what was the predecessor state
        self.machine = None
        self.successors = None #the successor states will be added when creating the entire tree of states
        
        #costs and actions
        self.costs = None #cost to transition from the predecessor state to the current one
        self.action = (None, None) #action to transition from the predecessor state to the current one
        self.transition_dic = {}
        
        #optimal future costs
            #will be added after all states were created
            #list of length n+1 (n is number oj jobs)
            #the entry on row i stands for the optimal future costs to assign job i to the chosen machine
            #last row stands for turning off a machine
        self.Qvalues = [None]*(n+1)
        
        #data for Neural Network
        self.input = None
        self.target = None


# ### Creation of all States
# We will now define all the necessary functions so that all the states can be created.

# In[3]:


#Create every possible state
def create_all_states(list_jobs,list_machines):
    
    #initiate list of states = []
    list_states = []
    
    #create initial state
    initial_state = create_initial_state()
    ID = 0
    
    #list of current states
    current_states = [initial_state]
    
    #go through every current state, save their successors, add current states to list of all states
        #then define successor states as current states, clear list of successor states and repeat until done
    while current_states:
        
        #empty list of all successor states
        successor_states = []
        
        #create and add all successor states of current states
        for state in current_states:
            
            #give ID
            state.ID = ID
            ID += 1
            
            #list of all successors of this state
            state_successors = []
            
            #check if we are already in a final state
            if sum(state.jobs_remaining) == 0:
                #add remaining costs until everything is finished, if that is not the case yet
                if state.machine_runtimes:
                    state.costs += max(state.machine_runtimes)
                
            #create one state for every remaining job assigned to the free machine with lowest index
            else:
                
                #get free machine with lowest index as object
                machine = list_machines[state.free_machines.index(1)]
                state.machine = machine.index
                
                #create data for Neural Network
                state_to_data(state)
                    
                #loop through jobs
                list_jobs_remaining = [job for job in list_jobs if state.jobs_remaining[job.index] == 1]
                for index, job in enumerate(list_jobs):
                    #check if the job still has to be done
                    if state.jobs_remaining[index] == 1:
                        #assign job to machine
                        state_successors.append(assign_job(state,job,machine))    
                
                #check if turning it off is an option
                if sum(state.machines_on_duty) > 1:                       
                    #add successor state created by shutting down machine
                    state_successors.append(turn_off_machine(state, machine))
                        
                
            #add successor list to the attributes of state
            state.successors = state_successors
            
            #add successors of this state to the list of all successors of all current states
            successor_states += state_successors
        
        #add current states to list of all states
        list_states += current_states
        
        #the successor states then become the current states
        current_states = successor_states
        
    return list_states


# In[1]:


#create the initial state
def create_initial_state():
    
    #create initial circumstances
    time = 0
    jobs_remaining = [1]*n
    machines_on_duty = [1]*m
    machine_runtimes = [machine.init_runtime for machine in list_machines]
    machine_occupations = [None if x==0 else -1 for x in machine_runtimes] # "-1" stands for an initial occupation
    free_machines = [1 if x==0 else 0 for x in machine_runtimes]
    
    #create initial state
    initial_state = states(time, jobs_remaining, machines_on_duty, free_machines, machine_runtimes, machine_occupations, None)
    initial_state.costs = initialize_costs(initial_state)
    
    return initial_state


# In[5]:


def assign_job(state,job,machine):
    
    #job gets canceled from to-do-list
    jobs_remaining = state.jobs_remaining.copy()
    jobs_remaining[job.index] = 0
    
    #machine is not free anymore
    free_machines = state.free_machines.copy()
    free_machines[machine.index] = 0
    
    #it gets a runtime equivalent to the jobs processing time
    machine_runtimes = state.machine_runtimes.copy()
    machine_runtimes[machine.index] = job.processing_time [machine.index]
    
    #we assign this job to the machine
    machine_occupations = state.machine_occupations.copy()
    machine_occupations[machine.index] = job.index 
    
    #if there is another free machine, we create a successor state at same time
    if sum(free_machines) > 0:
        #no time passes
        time_difference = 0
        #create state
        successor_state = states(state.time, jobs_remaining, state.machines_on_duty,
                                 free_machines, machine_runtimes, machine_occupations, state)
    
    #elsewise we have to proceed to the point where the next job finishes
    else:
        #calculate new time for when next job finishes
        time_difference = min([runtime for runtime in machine_runtimes if runtime > 0])
        new_time = state.time + time_difference
                            
        #the following machine(s) become free
        for index in range(m):
            if machine_runtimes[index] == time_difference:
                free_machines[index] = 1
                machine_occupations[index] = None
                            
        #update machine runtimes
        machine_runtimes = [max(runtime - time_difference,0) for runtime in machine_runtimes]
                            
        #create succesor state
        successor_state = states(new_time, jobs_remaining, state.machines_on_duty,
                                 free_machines, machine_runtimes, machine_occupations, state)
        
    #give its action and costs
    successor_state.action = (job.index,machine.index)
    successor_state.costs = transition_costs(state,successor_state,job,machine)

        
    #add successor state to transition dictionary
    state.transition_dic[(job.index,machine.index)] = successor_state    
    
    return successor_state


# In[6]:


#function to create all successors of a state by shutting down machine
def turn_off_machine(state, machine):
    
    #machine gets shut down
    machines_on_duty = state.machines_on_duty.copy()
    machines_on_duty[machine.index] = 0
                        
    #machine is not free anymore
    free_machines = state.free_machines.copy()
    free_machines[machine.index] = 0                              
                        
    #if there is another free machine, we create a successor state at same time
    if sum(free_machines) > 0:
        #no time passes
        time_difference = 0
        #create state
        successor_state = states(state.time, state.jobs_remaining, machines_on_duty,
                                 free_machines, state.machine_runtimes, state.machine_occupations, state)
        
    #elsewise we have to proceed to the point where the next job finishes
    else:
        #calculate new time for when next job finishes
        time_difference = min([runtime for runtime in state.machine_runtimes if runtime > 0])
        new_time = state.time + time_difference
                            
        #the following machine(s) become free
        machine_occupations = state.machine_occupations.copy()
        for index in range(m):
            if state.machine_runtimes[index] == time_difference:
                free_machines[index] = 1
                machine_occupations[index] = None
                            
        #update machine runtimes
        machine_runtimes = [max(runtime - time_difference,0) for runtime in state.machine_runtimes]
                            
        #create succesor state
        successor_state = states(new_time, state.jobs_remaining, machines_on_duty,
                                 free_machines, machine_runtimes, machine_occupations, state)
        
    #give its action and costs
    successor_state.action = (n,machine.index)
    successor_state.costs = transition_costs(state,successor_state,None,machine)
        
        
    #add successor state to transition dictionary
    state.transition_dic[(n,machine.index)] = successor_state
        
    return successor_state


# In[7]:


def initialize_costs(initial_state):
    
    costs = 0
    for i, runtime in enumerate(initial_state.machine_runtimes):
        machine = list_machines[i]
        costs += max(0,machine.weight*(runtime - machine.deadline))
        
    return costs


# In[8]:


def transition_costs(state, successor_state, job, machine):
    
    #runtime costs (time passing + deadline cost of idle jobs)
    st = state.time #start time
    nt = successor_state.time #next time
    transition_costs = nt - st #cost of time passing
    #jobs can be idle, therefore deadline costs might have to be payed while waiting for next state
    for i, job_i in enumerate(list_jobs):
        if successor_state.jobs_remaining[i] == 1: #check if job has been assigned already
            transition_costs += max(0, job_i.weight*(nt-max(job_i.deadline,st)))   
    
    #deadline costs (if job was assigned/state is not a machine turn-off state)
    if job:
        proc_time = job.processing_time[machine.index]
        ct = st + proc_time #completion time
        transition_costs += max(0,job.weight*(ct - max(job.deadline,st))) #job deadline overdue costs
        transition_costs += max(0,machine.weight*(ct - max(machine.deadline,st))) #machine deadline overdue costs

    return transition_costs


# ### Data
# 
# We want to turn the information of every state into a data type that we can feed into our Neural Network.
# To interprete the states as feedable Data, we need them to contain the following information accessible and compatible for a Neural Network:
# 
# - Matrix of Jobs (row-wise): Runtime per Machine, Time until Deadline and Weight.
# - Matrix of Machines (row-wise): Remaining Runtime of Machine, Time until Deadline and Weights.
# - Target: Vector of Q-Values. If an action is non-feasible it gets value zero and will be canceled out in the Neural Network later. Also the corresponding one-hot-vector indicating the optimal action.
# 
# Finally, we flat out the matrices and normalize the Runtimes by the given Maximal Processing Time, the times until Deadline by the Maximal Deadline and the Weights by the Maximal Weight.

# In[15]:


#take information of state to create normalized data for Neural Network
def state_to_data(state):
    
    #normalized information about the jobs
    jobs_data = np.asarray([[(proc_time/max_runtime) for i, proc_time in enumerate(job.processing_time)
                             if state.machines_on_duty[i] == 1]
                            + [max(job.deadline-state.time,0)/max_deadline,
                               job.weight/max_weight]
                            for job in list_jobs if state.jobs_remaining[job.index] == 1])
    
    #normalized information about the machines
    machines_data = np.asarray([[state.machine_runtimes[machine.index]/max_runtime,
                                 max(machine.deadline-state.time,0)/max_deadline,
                                 machine.weight/max_weight] 
                                for machine in list_machines if state.machines_on_duty[machine.index] == 1])
                             
    #sort them and save permutation
    machines_perm = machines_data[:,0].argsort() #sort machines by remaining runtime
    orig_order = np.arange(len(machines_perm)) #just an array of the form [0,1,...,m_state-1]
    jobs_data[:,orig_order] = jobs_data[:,machines_perm] #reorder processing time of jobs by new order of machines
    jobs_perm = jobs_data[:,0].argsort() #order of jobs by processing time for current free machine
    jobs_data = jobs_data[jobs_perm] #sort jobs by this order
    machines_data = machines_data[machines_perm] #sort machines by their remaining runtime
    
    #merge
    state.input = [jobs_data, machines_data]
    state.permutation = [jobs_perm,machines_perm]


# ### Q-Values
# We will now define the functions that compute the Q-Values for every action from every State.
# As soon as the Q-Values of a state are computed, we add them as as target vector, replacing the "None" -entries with zero to be compatible for the Neural Network later on, where these zero-values referring to non-feasible action will get cancelled out. We also add the corresponding one-hot-vector indication the optimal action.

# In[10]:


#find Q-values of every state by backtracking
def backtracking(all_states):
    
    #new list of all states that have already been backtracked completely
    backtracked_states = []
    #list of states that yet have to be backtracked
    states_to_backtrack = all_states.copy()
    
    for state in states_to_backtrack:
        #how many successors of this node have already been backtracked
        state.backtracking = - len(state.successors) #we will count upwards until zero
                
    while states_to_backtrack:
        #states that are temporary final in regard to backtracking (all their successors have backtracked already)
        temp_final_states = []
        
        #sort list, so we can pop all nodes that are temporary final in every step
        states_to_backtrack.sort(key=operator.attrgetter('backtracking'))
        for state in states_to_backtrack[::-1]:
            if state.backtracking == 0:
                temp_final_states.append(states_to_backtrack.pop())
            else:
                break
        
        #add values of temporary final states to the optimal future costs of their predecessors at its place for this action
        for state in temp_final_states:
            predecessor = state.predecessor
            #define target vector corresponding to Q-Values and One-Hot-Vector indicating optimal decision in permutaded order
            n_state = sum(state.jobs_remaining)
            if n_state > 0:
                #get Qvalues of all feasible actions
                target = np.array([qvalue for qvalue in state.Qvalues if qvalue != None])
                #sort by permutation
                target[np.arange(n_state)] = target[state.permutation[0]]
                state.target = [target, 
                                np.eye(target.shape[0])[np.argmin(target)], #one_hot_vector of optimal action
                               np.min(target)/target] #normalize by scaling through minimum value and then taking inverse value
            #stop when there is no predecessor anymore
            if not predecessor:
                break
            #add optimal future costs at position of action
            predecessor.Qvalues[state.action[0]] = state.costs + minimum(state)[0]
            #count upwards of how many successors this state still needs its Q-value before becoming temporary final itself
            predecessor.backtracking += 1    


# In[11]:


#function to get the minimum Qvalue and its belonging action from a state
def minimum(state):
    minimum = None
    job = None

    for i, val in enumerate(state.Qvalues):
        if val != None:
            if minimum == None:
                minimum = val
                job = i
            elif val < minimum:
                minimum = val
                job = i
    
    #final states have no successors, so we set all future costs to zero
    if minimum == None:
        minimum = 0
    
    action = (job,state.machine)
        
    return minimum, action


# ### Computation of all states
# We merge now all of our former results into one single function that shall compute all possible states together with all of their Q-Values.

# In[12]:


#the entire process from creation of all states to updating their Qvalues by backtracking
def compute_all_states(jobs, machines, max_r, max_d, max_w, MVS=0, JS=0, save=False, name="all_states"):
    
    #define global parameters for easier access
    global list_jobs, list_machines, n,m#, jobs_data, machines_data
    list_jobs, list_machines = jobs, machines
    n, m = len(jobs), len(machines)
    global max_runtime, max_deadline, max_weight
    max_runtime, max_deadline, max_weight = max_r, max_d, max_w
    
    #measure starting time
    st = time.time()
    
    #create all states
    all_states = create_all_states(list_jobs,list_machines)
    
    #update their Qvalues by backtracking
    backtracking(all_states)
    
    #save the computed states
    if save:
        path = f'{name}.pickle'
    else:
        MVS_str = "0"*(2-len(str(MVS))) + str(MVS)
        JS_str = "0"*(4-len(str(JS))) + str(JS)
        path = f'MaxValuesSets/MaxValues_{MVS_str}/JobScheduling_{JS_str}/all_states_{MVS_str}_{JS_str}.pickle'
    with open(path, 'wb') as f:
        pickle.dump(all_states, f, pickle.HIGHEST_PROTOCOL)
        
    
    #measure end time
    et = time.time()
    
    #tell how much time the entire process took
    print(round(et-st,2), "seconds to compute", len(all_states), "states.")
    
     return(all_states)


# ### Policies
# Having created all the states and the Q-Values corresponding to each of their actions, we can now create Policies.<br>
# A Policy is a tuple an consist of the list of all actions and a list of all States.

# In[13]:


#compute random policy
def random_policy(all_states):
    
    #initial state
    rand_state = all_states[0]
    
    random_actions = []
    random_states = [rand_state]
    
    while rand_state.successors:
        #choose a random successor state
        rand_successor = random.choice(rand_state.successors)
        #add this random action and state to policy
        random_actions.append(rand_successor.action)
        random_states.append(rand_successor)
        #continue policy from this state
        rand_state = rand_successor
        
    return random_actions, random_states


# In[14]:


#compute optimal policy
def optimal_policy(all_states, name="optimal_policy"):
    
    initial_state = all_states[0]
    optimal_actions = []
    optimal_states = []
    state = initial_state
    
    #go down the tree, always choosing the successor with the minimal Q-value
    while state.successors:
        #add state
        optimal_states.append(state)
        #add action
        optimal_action = minimum(state)[1]
        optimal_actions.append(optimal_action)
        #declare the successor
        for successor in state.successors:
            if successor.action == optimal_action:
                state = successor
    
    #add final state            
    optimal_states.append(state)
    
    #safe optimal policy
    with open(f'{name}.pickle', 'wb') as f:
        pickle.dump((optimal_actions,optimal_states), f, pickle.HIGHEST_PROTOCOL)
    
    return optimal_actions,optimal_states

