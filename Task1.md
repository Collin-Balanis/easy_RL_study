_**2022-7-11**_    
_**collin**_    
**task1 of easy_RL**    
# Chapter1 强化学习概述    
## 1.1强化学习概述
**强化学习（reinforcement learning，RL）** 讨论的问题是智能体（agent）怎么在复杂、不确定的环境（environment）中最大化它能获得的奖励。核心问题对应的框图是下图。
<div align=center>
<img width="550" src="https://github.com/Collin-Balanis/easy-rl/blob/master/docs/chapter1/img/1.1.png"/>
</div>
<div align=center>图 1.1 强化学习示意（引用教程图片）</div>
教程中还重点强调了**强化学习与监督学习**的区别。 列举了一些强化学习应用的例子。
## 1.2序列决策##     
智能体与环境是相互交互的。 智能体给环境输入一个动作，环境根据动作做出下一步的观测与奖励。每一步与上一步是有关系的。无法做到监督学习那样的独立同分布。
### 奖励     
奖励是环境给智能体的一个反馈。
### 序列决策     
在与环境的交互过程中，智能体会获得很多观测。针对每一个观测，智能体会采取一个动作，也会得到一个奖励。所以历史是观测、动作、奖励的序列：
 $$ H_{t}=o_{1}, r_{1}, a_{1}, \ldots, o_{t}, a_{t}, r_{t} $$
智能体在采取当前动作的时候会依赖于它之前得到的历史，所以我们可以把整个游戏的状态看成关于这个历史的函数：
 $$ S_{t}=f(H_{t})$$
## 1.3动作空间     
分为有限离散的空间和连续动作空间。
## 1.4强化学习组成部分     
* **策略** 用策略选下一步的动作      
**随机性策略** 就是$\pi$函数,即$\pi(a | s)=p\left(a_{t}=a | s_{t}=s\right)$。输入一个状态$ s $,输出一个概率。
**确定性策略** 就是智能体直接采取最有可能的动作，即$a^{*}=\underset{a}{\arg \max} \pi(a \mid s)$

* **价值函数** 价值函数对当前状态进行评估。
价值函数的值是对未来奖励的预测，我们用它来评估状态的好坏。
价值函数里面有一个**折扣因子（discount factor）**，我们希望在尽可能短的时间里面得到尽可能多的奖励。

$$V_{\pi}(s) \doteq \mathbb{E}_{\pi}\left[G_{t} \mid s_{t}=s\right]=\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^{k} r_{t+k+1} \mid s_{t}=s\right],\text{对于所有的} s \in S$$

还有一种价值函数：Q 函数。

$$Q_{\pi}(s, a) \doteq \mathbb{E}{\pi}\left[G{t} \mid s_{t}=s, a_{t}=a\right]=\mathbb{E}{\pi}\left[\sum{k=0}^{\infty} \gamma^{k} r_{t+k+1} \mid s_{t}=s, a_{t}=a\right]$$

* **模型** 表示智能体对环境状态的理解

下一步的状态取决于当前的状态以及当前采取的动作。
它由状态转移概率和奖励函数两个部分组成。状态转移概率即
$$p_{s s^{\prime}}^{a}=p\left(s_{t+1}=s^{\prime} \mid s_{t}=s, a_{t}=a\right)$$
奖励函数是指我们在当前状态采取了某个动作，可以得到多大的奖励，即 
$$R(s,a)=\mathbb{E}\left[r_{t+1} \mid s_{t}=s, a_{t}=a\right]$$
当我们有了策略、价值函数和模型3个组成部分后，就形成了一个**马尔可夫决策过程（Markov decision process）。**

