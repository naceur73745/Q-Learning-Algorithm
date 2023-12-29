from Agent  import Agent 
from env import Prisoners 
import numpy as np 


def Create_agent(input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size):
  return Agent(input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size)


def train_function(episode_len  , n_games ,input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size ):
  env = Prisoners(episode_len,n_games)
  agent  = Create_agent(input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size )
  for Round in range  (n_games ):
    state = env.reset()
    step = 0
    while env.done == False   :
      action = agent.choose_action(state)
      new_state  , reward , done , info =  env.step(action)
      #agent.mem.store_action(state , new_state,action ,reward ,done  )
      agent.learn(state, action, reward  , new_state )
      state = new_state

      print( f" Round  : {Round} , step:{step} ")
      step+= 1
    agent.epsilon = agent.min_epsilon + (agent.max_epsilon - agent.min_epsilon)*np.exp(-agent.decay_rate*Round)

  return env.state_total , env.reward_total , env.chooosed_startegy_each_round ,agent


def plot_Reward(reward_over_epochen , title  , label , x_label  , y_label ):


    epochs = range(1, len(reward_over_epochen) + 1)
    plt.plot(epochs, reward_over_epochen, 'b', label= label  )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

"""#Train Agent"""

import matplotlib.pyplot as plt



#2,128,512,2,0.0001,128,2048,0.99,0.4,0.2,0.95
def Create_agent(input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size):
  return Agent(input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size)

def train_function(episode_len  , n_games ,input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size ):
  rewards= []
  env = Prisoners(episode_len,n_games)
  agent  = Create_agent(input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size )
  for Round in range  (n_games ):
    state = env.reset()
    step = 0
    rewards_pro_Round= []
    while env.done == False   :
      action = agent.choose_action(state)
      new_state  , reward , done , info =  env.step(action)
      rewards_pro_Round.append(reward)
      #agent.mem.store_action(state , new_state,action ,reward ,done  )
      agent.learn(state, action, reward  , new_state )
      state = new_state


      print( f" Round  : {Round} , step:{step} ")
      step+= 1
    rewards.append(np.mean(rewards_pro_Round))

    agent.epsilon = agent.min_epsilon + (agent.max_epsilon - agent.min_epsilon)*np.exp(-agent.decay_rate*Round)

  return env.state_total , env.reward_total , env.choosen_startegy_each_round , rewards, agent
#def train_function(episode_len  , n_games ,input_dimension , n_actions ,  lr , gamma  , epsilon , max_epsilon , min_epsilon  , decay_rate, mem_size , butch_size ):







states  , reward_tuple  , startegies,reward_ai , agent   = train_function(1000 , 50 , 4,2,0.8,0.95,1.0 , 1.0 , 0.9 , 0.005 , 1024 , 32 )
print( "train mode ")
print(states )
print(reward_tuple )
print(startegies )
print("\n")
#print(f"Qtable  : {agent.Q_table}")



plot_Reward(reward_ai , "reward over epochen "  , "somthiing" , "epochen"  , "rewards" )

"""
#Function for Evaluation 's
"""

#function for evaluation
#this function plotts the cooperation rate and defection rate for a player
def plot_bar_chart(rate_of_cooperation, rate_of_defection, title_under_first_plot, title_under_second_plot, title, y_label):
    x = [1, 2]  # x-axis values

    # Heights of the bars
    heights = [rate_of_cooperation, rate_of_defection]

    # Labels for the bars
    labels = [title_under_first_plot, title_under_second_plot]

    # Plotting the bar chart
    plt.bar(x, heights)

    # Adding labels and title
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x, labels)

    # Displaying the plot
    plt.show()


#this  calculates the rewards  of player the reward list should be for only one round

def calculate_reward(reward_list , player_index ):

  reward  = 0
  for state in  reward_list :
    reward+= state[player_index ]
  return reward


#this function used to define the cooperation rate and  and defction rate for player in round and it works only on oponent who alw coop and defct

def coop_defect_rate (state_list ,strategy  ) :

  #we cann add the others startegy also  to test
  right_decision =  0
  false_decision = 0
  for state  in state_list   :

   if strategy == 1: #stands for opent is always defecting  the right decison  will be (1,1)
      if state ==(1,1) :
        right_decision = right_decision+1
      else :
        false_decision = false_decision+1
   else :

      if state ==(1,0) : #in case the oponent always coop  the right decison will eb to alwys defect

        right_decision = right_decision+1
      else :
        false_decision = false_decision+1

  return right_decision , false_decision, strategy

#create an  agent
  def Create_agent(input_dim ,dim1 , dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda):
    return Agent(input_dim ,dim1, dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda)
#this plott rewqrds over epochen

#plott reword over epochen

def plot_Reward(reward_over_epochen1,reward_over_epochen2  , title  , label1, label2 , x_label  , y_label ):


    epochs = range(1, len(reward_over_epochen1) + 1)
    plt.plot(epochs, reward_over_epochen1, 'b', label= label1 )
    plt.plot(epochs, reward_over_epochen2, 'r', label= label2  )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()



def test_function (epsiode_len , n_games ,values, agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]

  for Round in  range (n_games ):

    state = env.reset()

    for i in range  (epsiode_len ) :

      state  = agent.convert_function(state )

      action = np.argmax(agent.Q_table[state,:])

      new_state  , reward , done , info =  env.evaluate(action ,values  )

      states.append(new_state)

      rewards.append(info )

      state = new_state

  return states  , rewards




#count defect coooperation rate
def count_coop_defect_rate (list_of_state):
      player1_defection_rate = 0
      palyer1_cooperation_rate= 0
      #player 2
      player2_defection_rate = 0
      palyer2_cooperation_rate= 0

      for state in list_of_state :
        if state[0] == 1  :
          player1_defection_rate+= 1
        else  :
          palyer1_cooperation_rate+= 1

        if state[1] == 1  :
          player2_defection_rate+= 1
        else  :
          palyer2_cooperation_rate+= 1
      return  player1_defection_rate , palyer1_cooperation_rate  , player2_defection_rate , palyer2_cooperation_rate


#plott
def generate_bar_chart(player1_rates, player2_rates):

    # Extract cooperation and defection rates for Player 1
    player1_cooperation_rate = player1_rates[0]
    player1_defection_rate = player1_rates[1]

    # Extract cooperation and defection rates for Player 2
    player2_cooperation_rate = player2_rates[0]
    player2_defection_rate = player2_rates[1]

    # Bar positions for Player 1 and Player 2
    player1_positions = [0, 1]
    player2_positions = [3, 4]

    # Heights of the bars
    player1_heights = [player1_cooperation_rate, player1_defection_rate]
    player2_heights = [player2_cooperation_rate, player2_defection_rate]

    # Bar labels
    player1_labels = ['Cooperation', 'Defection']
    player2_labels = ['Cooperation', 'Defection']

    # Plotting the bar chart
    plt.bar(player1_positions, player1_heights, align='center', alpha=0.5, label='Player 1')
    plt.bar(player2_positions, player2_heights, align='center', alpha=0.5, label='Player 2')

    # Adjusting the x-tick positions and labels
    plt.xticks([0, 1, 3, 4], player1_labels + player2_labels)

    plt.ylabel('Rate')
    plt.title('Cooperation and Defection Rates')

    # Adding legend
    plt.legend()

    # Display the chart
    plt.show()

"""#Evalutate The Agent's

"""

import matplotlib.pyplot as plt



def test_function (epsiode_len , n_games ,values, agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]
  rewards_pro_player = []
  player_reward2= []
  for Round in  range (n_games ):
    state = env.reset() #state randomly
    sum = 0
    sum2=.0
    for i in range  (epsiode_len ) :

      state  = agent.convert_function(state )

      action = np.argmax(agent.Q_table[state,:])
      new_state  , reward , done , info =  env.evaluate(action ,values  )
      sum+=reward
      sum2+=info[1]
      states.append(new_state)

      rewards.append(info )
      rewards_pro_player.append(sum)
      player_reward2.append(sum2)
      state = new_state

  return states  , rewards ,rewards_pro_player ,player_reward2

eval_states , eval_rewards ,player_reward , player_reward2= test_function(100 , 1 ,0  , agent)


print("**********************************Test ******************************\n")

print(f"the current states  : {eval_states}")
print(f"the current rewards : {eval_rewards}")

print(f"\n")

print(f"\n")
print(f"the Graph for Reward over Epochen ")
print(f" eval  reward : {eval_rewards}" )
plot_Reward(player_reward ,player_reward2 , "reward over epochen " , " player1" , "player2" ,"epochen" , "reward")
print(f"\n")

print(f" defection cooperation rate  pro players\n")
player1_def  ,player1_coop , player2_def , player2_coop  = count_coop_defect_rate(eval_states)
print(f"player1_def  :{player1_def}")
print(f"player1_coop : {player1_coop}")
print(f" player2_def : {player2_def}")
print(f"player2_coop :{player2_coop}\n")

print(f" the bar char for cooperation and defection  Rate for each player ")

generate_bar_chart([player1_def,player1_coop ] , [player2_def , player2_coop])
