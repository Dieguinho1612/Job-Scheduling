#!/usr/bin/env python
# coding: utf-8

# # Visualizations

# ## Description of Notebook
# In this Notebook we will provide methods to visualize the States and Policies.<br>
# <br>
# We will start with two ways of describing a State:
#    1. Explaining it with words
#    2. Plotting it as a node of a graph together with its successor nodes
# 
# We will then provide three approaches to visualize a Policy:
#    1. Explaining it with words
#    2. Plot the entire Policy as a graph
#    3. Plot a gantt chart

# ## Code

# In[1]:


#import dependencies
import operator
import numpy as np
import pydot
from IPython.display import Image, display
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
#import import_ipynb
from Jobs_and_Machines import *


# ### Visualize States

# #### 1.: Explain a state with words

# The States are written in a way that a Neural Network can comprehend them.<br>
# We now write a helper function that actually explains the states with words and sentences easily understandable for the user.

# In[2]:


def explain_state(state):
    
    #set environment
    list_jobs, list_machines = state.jobs, state.machines
    n, m = len(list_jobs), len(list_machines)
    
    #what changed since the former state
    if state.predecessor:
        #check if machine got turned off
        if state.action[0] == n:
            print(f"Machine {state.action[1]+1} got shut down in the former state.")
        else:
            print(f"Job {state.action[0]+1} got assigned to machine {state.action[1]+1} in the former state.")
        #give time difference
        print(f"Since then {state.time - state.predecessor.time} time has passed.")
        #print costs
        print(f"The costs have been {state.costs}")
        print("")
    
    #about the jobs
    print("Remaining Jobs:", [list_jobs[i].number for i in range(n) if state.jobs_remaining[i] == 1])
    
    #about the machines
    print("The following machines are still on duty", [list_machines[i].number for i in range(m) if state.machines_on_duty[i] == 1])
    print("These machines are free:", [list_machines[i].number for i in range(m) if state.free_machines[i] == 1])
    print("The remaining processing time of occupied machines are:", [f"Machine {list_machines[i].number} Job {state.machine_occupations[i]+1} Time {state.machine_runtimes[i]}" for i in range(m) if (state.machines_on_duty[i] - state.free_machines[i]) == 1])
    
    #show time
    print("The time is ", state.time)
    #check if final state
    if sum(state.jobs_remaining) == 0:
        print("The final time is ", state.time + max(state.machine_runtimes))
    
    #about the other states
    print("The Predecessor State is", state.predecessor, "and the number of succesor states is", len(state.successors))
    
    print("")
    print("")


# #### 2.: Plot State as node of a graph
# For this we will
#    - create the node
#    - define the information it displays
#    - enbed it into a graph that also shows its successor nodes

# In[3]:


#create node for graph
    #1. green if final state
    #2. yellow if its the root/head state
    #3. brown if machine got shut down
    #4. blue else

def create_node(state,root=False):
    
    #set environment
    n = len(state.jobs)
    
    #check if final state
    if len(state.successors) == 0:
        color = "lawngreen"
    #check if root node
    elif root:
        color = "gold"
    #check if machine got turned down
    elif state.action[0] == n:
        color = "sandybrown"
    else:
        color = "lightblue"
    
    #create node    
    node = pydot.Node(node_designation(state,root), color=color, style="filled")
    
    return node


# In[4]:


#we define now which information the nodes will show
def node_designation(state, root=False):
    
    #set environment
    list_jobs, list_machines = state.jobs, state.machines
    n, m = len(list_jobs), len(list_machines)
    
    #name
    name = "ID" + str(state.ID)
    
    if root == False:
        #check if machine got turned off
        if state.action[0] == n:
            name += f"\nMachine {state.action[1]+1} shut down"
        else:
            name += f"\nJob {state.action[0]+1} to Machine {state.action[1]+1}"
        #costs
        name += f"\nCosts = {state.costs}"
        #give time difference
        name += f"\nTime Delta = {state.time - state.predecessor.time}\n"
    
    #time
    name += f"\nTime = {state.time}"
        
    #if it is not a final state, we also print all the information about the remaining jobs and the machine
    if sum(state.jobs_remaining) > 0:
        #remaining jobs
        name += "\nJobs remaining = " + ", ".join([str(list_jobs[i].number) for i in range(n) if state.jobs_remaining[i] == 1])
        #machines
        name += "\nMachines on duty = " + ", ".join([str(list_machines[i].number) for i in range(m) if state.machines_on_duty[i] == 1])
        name += "\nFree Machines = " + ", ".join([str(list_machines[i].number) for i in range(m) if state.free_machines[i] == 1])
        #if any machine s+is currently working on a job, print the runtimes
        if sum(state.machine_runtimes) > 0:
            name += "\nMachine Runtimes = " + " || ".join(["M " + f"{list_machines[i].number} T {state.machine_runtimes[i]}" for i in range(m) if (state.machines_on_duty[i] - state.free_machines[i]) == 1])
        else:
            name += "\nMachine Runtimes = /"
        
    #in case of a final state, print instead the costs and the time until everything will be finished
    else:
        #Time until the End
        name += f"\nTime to End = {max(state.machine_runtimes)}\n"
        #Time at End
        name += f"\nFinal Time = {state.time + max(state.machine_runtimes)}"
        #give final costs
        final_costs = state.costs
        predecessor = state.predecessor
        while predecessor:
            final_costs += predecessor.costs
            predecessor = predecessor.predecessor
        name += f"\nFinal Costs = {final_costs}"   
        
        
    return name


# In[5]:


#create partial graph, consisting of any node and its successors
def partial_graph(state):
    
    #create the graph
    G_part = pydot.Dot(graph_type = "digraph")
    
    #add state as root
    node = create_node(state, True)
    G_part.add_node(node)
    
    #add all successors as nodes
    for successor in state.successors:
        node = create_node(successor)
        G_part.add_node(node)
        edge = pydot.Edge(node_designation(state,True),node_designation(successor))
        G_part.add_edge(edge)
    
    #plot graph
    display(Image(G_part.create_png()))


# ### Visualize Policy

# #### 1.: Explain Policy with words

# In[6]:


#describe policy
def explain_policy(policy):
        
    #the input is a tuple consisting of a list of actions and a list of states
    actions = policy[0]
    states = policy[1]
    n = len(states[0].jobs)
    
    #for every action, show time and explain what happens
    for i, action in enumerate(actions):
        print(f"Time {states[i].time}:")
        #check if machine got turned off
        if action[0] == n:
            print(f"Machine {action[1]+1} gets shut down")
        #else job got assigned to machine
        else:
            print(f"Job {action[0]+1} gets assigned to Machine {action[1]+1}")
        print("")
    
    #add final state
    final_state = states[-1]
    print(f"The schedule terminates at time {final_state.time + max(final_state.machine_runtimes)}")        


# #### 2.: Plot Policy as a graph

# In[7]:


#draw graph of policiy
def draw_policy_graph(policy):
    
    #a policy is a tuple of the actions and state, but here we only need the states
    states = policy[1]
    
    #create graph
    G_opt = pydot.Dot(graph_type = "digraph")
    
    #create root
    node = create_node(states[0],True)
    G_opt.add_node(node)
    
    #create node for every state in policy
    for state in states[1:]:
        node = create_node(state)
        G_opt.add_node(node)
        #technical distinction for when the predecessor is the root
        if state.predecessor.predecessor:
            edge = pydot.Edge(node_designation(state.predecessor),node_designation(state))
            G_opt.add_edge(edge)
        else:
            edge = pydot.Edge(node_designation(state.predecessor,True),node_designation(state))
            G_opt.add_edge(edge)
    
    #plot graph
    display(Image(G_opt.create_png()))


# #### 3.: Plot Policy as a gantt chart
# The Gantt Chart represents a Policy. The incurred costs get displayed as well.<br>

# Every bar in the gantt chart stands for a Job. It has a hover displaying all the corresponding information:<br>
# 
#    - When the job got started
#    - How long its processing time was
#    - When it finished
#    - Its Deadline
#    - How late we were finishing it regarding its deadline
#    - What (penalty) weight it has
#    - How much costs exceeding its deadline generated

# For some Machines, there is a State whose action is to turn it off.<br>
# However, if a Machine is still working during the assignment of the last Job, no additional shut down State will be added because the deed of doing so is already logically inherent.<br>
# Nevertheless, in the gantt chart we add a black bar at the end of every Machine-timeline, standing for the action of turning it off and containing a hover with the following information:
# 
#    - When did the machine got shut down
#    - The Deadline of the machine
#    - How late we were turning it off regarding its deadline
#    - What (penalty) weight it has
#    - How much costs exceeding its deadline generated

# In the title there are also given the total costs if the job schedule. It consists of three parts:
# 
#    - The runtime costs: For this just take the latest finishinig time of a job
#    - The deadline costs of the jobs: Sum over all the deadline costs of the jobs (displayed in their hovers)
#    - The deadline costs of the machines: Sum over all the deadline costs of the machines (displayed in their hovers)
#    
# The sum of these three parts result in the Total Costs.

# Last, in some cases the Machines are occupated initially, which will be represented by a grey bar.

# In[8]:


#print policy as gantt chart
def draw_gantt_chart(policy):
    
    #filter for states in which jobs got assigned
    initial_state = policy[1][0] 
    n = len(initial_state.jobs)
    states = [state for state in policy[1][1:] if state.action[0] < n]
    
    #we add each state as the assignment of a job to a machine
    list_jobs, list_machines = states[0].jobs, states[0].machines
    job_assignments = []
    for state in states:
        #get job and machine
        job_ind, machine_ind = state.action
        job, machine = list_jobs[job_ind], list_machines[machine_ind]
        proc_time = job.processing_time[machine_ind]
        time = state.predecessor.time
        #add to gantt chart
        job_assignments.append(dict(Resource=f"Machine {machine.number}", 
                                Start=time, 
                                Finish=time + proc_time,
                                Runtime=proc_time,
                                Task=f"Job {job.number}",
                                Deadline=job.deadline,
                                Lateness=max(0,time + proc_time - job.deadline),
                                Weight=job.weight,
                                Costs=job.weight * max(0,time + proc_time - job.deadline)
                                   ))
    
    
    #We now also want to determine when the machines get turned off
    #This includes the logically inherent shut down of machines which are not listed as additional states
    machine_shut_downs = []
    #add shut down action for every machine
    for machine in list_machines:
        #find out last time machine was assigned
        #"state" will actually be the successor state of that, since "action" refers to what happened in the predecessor state
        for state in policy[1][::-1]:
            #check if state is already a shut down state
            if state.action[1] == machine.index:
                if state.action[0] == n:
                    #in this case time of predecessor state is time of shut down
                    time=state.predecessor.time
                #else take the time of the last assignment of the machine and add the processing time
                else:
                    job = list_jobs[state.action[0]]
                    time = state.predecessor.time + job.processing_time[machine.index]
                
                #add to gantt chart
                machine_shut_downs.append(dict(Resource=f"Machine {machine.number}", 
                                        Start=time, 
                                        Finish=time,
                                        Runtime=0,
                                        Task="Shut Down",
                                        Deadline=machine.deadline,
                                        Lateness=max(0,time - machine.deadline),
                                        Weight=machine.weight,
                                        Costs= machine.weight * max(0,time - machine.deadline)
                                              ))
                break
    
    #now we also have to add the initial runtimes
    initial_runtimes = [dict(Resource=f"Machine {i+1}", 
                                        Start=0, 
                                        Finish=initial_runtime,
                                        Runtime=initial_runtime,
                                        Task="Initial Occupation",
                                        Deadline=initial_runtime,
                                        Lateness=0,
                                        Weight=0,
                                        Costs= 0)
                        for i,initial_runtime in enumerate(initial_state.machine_runtimes)]
    
    #the timeline lists all actions (jobs get sorted for legend) 
    timeline = initial_runtimes + sorted(job_assignments, key=lambda dic: dic['Task']) + machine_shut_downs
    
    #create data frame
    df = pd.DataFrame(timeline)
    df['Delta'] = (df['Finish'] - df['Start']).clip(0.2) #"clip" gives the Shut-Down Bar a width of 0.2
    
    #create figure
    fig = px.bar(df, base="Start", x="Delta", y="Resource", orientation="h", color="Task",
                 hover_data={"Task":False, "Delta":False, "Resource":False, "Runtime":True, "Finish":True,
                             "Deadline":True, "Lateness":True, "Weight":True, "Costs":True}, 
                             color_discrete_map={"Initial Occupation": "lightgrey", "Job 1": "dodgerblue", "Shut Down":"black"},
                title="Gantt Chart of Job Schedule<br>Total Costs " + str(np.sum(df["Costs"]) + np.max(df["Finish"])))
    
    #create axes
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Machine", autorange="reversed")
    
    #add bar description
    x,y,text = df["Start"] + 0.5*df["Delta"], df["Resource"], df.Task.where(~df.Task.isin(["Initial Occupation","Shut Down"])) #text pos. in middle of bar
    fig.add_trace(go.Scatter(x=x, y=y, text=text, textposition="middle center", #add text "Job X" to bar corresponding to Job X
                             mode="text", showlegend=False, hoverinfo="skip"))
    
    #draw gantt chart
    fig.show()

