#!/usr/bin/env python
# coding: utf-8

# # Classes of Jobs and Machines

# ## Description of Notebook
# In this Notebook we will create the Classes of Jobs and Machines for the Job Scheduling Problem in the deterministic case.

# ## Code

# In[9]:


#import dependencies
import operator
import pickle


# ### Machines

# We define the Class for Machines.<br>
# Every machine only consists of the information of its deadline, the belonging penalty weight if this deadline should be exceeded and the running time at which it is innitiated<br>

# In[2]:


class machines:
    def __init__(self, init_runtime, deadline, weight):
        self.number = None
        self.index = None
        self.init_runtime = init_runtime
        self.deadline = deadline
        self.weight = weight


# We want to sort them by their weights in descending order. In case of equal weights, they get sorted by their deadline.
# We then give every machine an index, so they are easier to call later on.<br>
# They will also receive a number, which is just the index increased by one, so the reader does not get confused with starting to count at zero when looking at plots and other types of visualizations.

# In[2]:


#prepare machines by sorting and indexing them
def prep_machines(list_machines, name="list_machines", save=False, prnt=False):
    
    #sort machines for weights in descending order or by their deadline in case of equal weights
    list_machines.sort(key=operator.attrgetter('deadline'))
    list_machines.sort(key=(operator.attrgetter('weight')), reverse=True)
    
    #give them a number and an index so they can be recognized more easily
    for i, machine in enumerate(list_machines):
        machine.index = i
        machine.number = i+1
        #print them to show the new order
        if prnt:
            print(f"Machine {machine.number}: Initial Runtime {machine.init_runtime} | Deadline {machine.deadline} | Weight {machine.weight}")
    
    #save list of machines
    if save:
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(list_machines, f, pickle.HIGHEST_PROTOCOL)
    
    return(list_machines)


# ### Jobs

# We now define the Class for Jobs.<br>
# Similiar to the machines, every job contains the information of its deadline and the belonging penalty weight if this deadline should be exceeded.<br>
# In addition it also needs a list of processing times, with each entry indicating how long the indexed machine needs to process this job.

# In[4]:


class jobs:
    def __init__(self,processing_time, deadline, weight):
        self.number = None
        self.index = None
        self.processing_time = processing_time
        self.deadline = deadline
        self.weight = weight


# Again, we will sort them by their weight in descending order (or by their deadline in case of equal weights), as well as index them and give them a number (equal to the index increased by one).

# In[8]:


#prepare jobs by sorting and indexing them
def prep_jobs(list_jobs, name="list_jobs", save=False, prnt=False):
    
    #sort jobs for weights in descending order or by their deadline in case of equal weights
    list_jobs.sort(key=operator.attrgetter('deadline'))
    list_jobs.sort(key=operator.attrgetter('weight'), reverse=True)
    
    #give them a number and an index so they can be recognized more easily
    for i, job in enumerate(list_jobs):
        job.index = i
        job.number = i+1
        #print them to show the new order
        if prnt:
            print(f"Job {job.number}: Processing Times {job.processing_time} | Deadline {job.deadline} | Weight {job.weight}")
        
    #save list of jobs
    if save:
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(list_jobs, f, pickle.HIGHEST_PROTOCOL)
    
    return(list_jobs)

