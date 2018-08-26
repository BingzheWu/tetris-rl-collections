# T-Agent: A Pytorch Library for Developing AI of Tetris Game

This is a general library for developing AI of Tetris Game. Our library
is built on Pytorch. The goal of this project is to ease 
the developemnt of Tetris AI using deep reinforcement learning algorithms.
One can easily integrated recent research progresses into their own AI
wthin our library. The Tteris simulator used in our project is from $https://github.com/jaybutera/tetrisRL$. Note that one can replace this simulator
with other gym-like simulator. 

## Motivation of this project
Firstly, we chose the Tetris game because it is simple yet interesting game.
All of the game information can be obtained from the scren, thus, we believe
that we can use a CNN to extract those informative features to help us train
the AI. As a result, we chose to develop Tetris AI in our final project. A major contribution of our project is a flexible RL framework that will be
introduced in the following section. 

We note that there are already a large number of public-available projects
to support different deep reinforcement learning algorithms. However, most
of these code repositories have servral following issues when applying to 
your own research:
  
  *    Hyper-parameters management system: There is always 
         a large number of hyper-parameters to adjust in training an RL based
         AI. Most of existed projects that built on Python used $argsparse$
         libary to setting the hyper-parameters. Thus, users can change the 
         parameters setting without changing the core codes. However, you need 
         to save these hyper-parameters as a file manually. In this project, 
         we propose to use YAML file to store all the parameters you need to run your program. Users need to write a YAML configuration before
         starting their experiment. An example of configuration YAMl file 
         can be found in ...
  
  *    Tight coupling of different modules: There are different distinct 
        components in a DRL-based agent, such as the enviroment, the neural
        network module and agent control part. We note that there is always
        a tight coupling of these different modules in previous projects.
        For example, it is always challenging to disentangle the enviroment 
        module and agent contro logic. Thus, users have to rewrite 
        the codes when they want to change their appliction, for example, from
        GYM env to user-difiend env. In this project, we try to decouple those
        distinct modules. Details can be found in the code structure part.

## Dependencies

 * Python3
 * Pytorch >= v4.0
 * Tetris Simulator

## Useage

## Code Structure
Here we give a brief overview of our whole projects. 
#### Config
This module contains the implementations of loading YAMl file to argument . The example
of configuration file can be found in. We will add more functions in the future, such as supportting the merge of our parameter system and the argparse
system.  

#### Network
This module contains the implementtations of different model architectures.
If you want to add new network architecture, you can write the nn module 
in Pytorch and directly add it into this module.

#### Replay Memory
This module contains the implementations of replay memory. We will add the
support of prioritied mechanism soon. This module is always for various DQN
variants and can be used for boost the computing efficiency of GPU.

#### Agent

This is the core module to make above modules cooperate togather. 
We firstly implement base class of Agent in .... The basic class contains
some common-use functions during the trainning of DRL algorithms. Generally
speaking, one agent instiation correspods with one specif algorithm, such DQN,
Double DQN and others. One can implement the specific agent by inherit the basic class. 


