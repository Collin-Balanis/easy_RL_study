_**2022-7-12**_    
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

## 1.2序列决策   
                                                                             
智能体与环境是相互交互的。 智能体给环境输入一个动作，环境根据动作做出下一步的观测与奖励。每一步与上一步是有关系的。无法做到监督学习那样的独立同分布。
### 奖励     
奖励是环境给智能体的一个反馈。
### 序列决策     
在与环境的交互过程中，智能体会获得很多观测。针对每一个观测，智能体会采取一个动作，也会得到一个奖励。所以历史是观测、动作、奖励的序列：
 $$H_{t}=o_{1}, r_{1}, a_{1}, \ldots, o_{t}, a_{t}, r_{t}$$
智能体在采取当前动作的时候会依赖于它之前得到的历史，所以我们可以把整个游戏的状态看成关于这个历史的函数：
 $$S_{t}=f(H_{t})$$
 
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

* **强化学习智能体的类型**
#### 1.基于价值的智能体与基于策略的智能体

根据智能体学习的事物不同，我们可以把智能体进行归类。**基于价值的智能体（value-based agent）** 显式地学习价值函数，隐式地学习它的策略。策略是其从学到的价值函数里面推算出来的。**基于策略的智能体（policy-based agent）** 直接学习策略，我们给它一个状态，它就会输出对应动作的概率。基于策略的智能体并没有学习价值函数。把基于价值的智能体和基于策略的智能体结合起来就有了**演员-评论员智能体（actor-critic agent）**。这一类智能体把策略和价值函数都学习了，然后通过两者的交互得到最佳的动作。

**教程里特别强调了基于策略和基于价值的强化学习方法的区别！！！**
> Q: 基于策略和基于价值的强化学习方法有什么区别?

> A: 对于一个状态转移概率已知的马尔可夫决策过程，我们可以使用动态规划算法来求解。从决策方式来看，强化学习又可以划分为基于策略的方法和基于价值的方法。决策方式是智能体在给定
> 状态下从动作集合中选择一个动作的依据，它是静态的，不随状态变化而变化。
> 在基于策略的强化学习方法中，智能体会制定一套动作策略（确定在给定状态下需要采取何种动作），并根据这个策略进行操作。强化学习算法直接对策略进行优化，使制定的策略能够获得最大> 的奖励。
> 而在基于价值的强化学习方法中，智能体不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作。基于价值迭代的方法只能应用在不连> 续的、离散的环境下（如围棋或某些游戏领域），对于动作集合规模庞大、动作连续的场景（如机器人控制领域），其很难学习到较好的结果（此时基于策略迭代的方法能够根据设定的策略来选> 择连续的动作）。
> 基于价值的强化学习算法有Q学习（Q-learning）、 Sarsa 等，而基于策略的强化学习算法有策略梯度（Policy Gradient，PG）算法等。此外，演员-评论员算法同时使用策略和价值评估来
> 做出决策。其中，智能体会根据策略做出动作，而价值函数会对做出的动作给出价值，这样可以在原有的策略梯度算法的基础上加速学习过程，取得更好的效果。
#### 2.有模型强化学习智能体与免模型强化学习智能体

**有模型（model-based）** 强化学习智能体通过学习状态的转移来采取动作。     
**免模型（model-free）** 强化学习智能体没有去直接估计状态的转移，也没有得到环境的具体转移变量，它通过学习价值函数和策略函数进行决策。免模型强化学习智能体的模型里面没有环境转移的模型。

用马尔可夫决策过程来定义强化学习任务，并将其表示为四元组 $<S,A,P,R>$ ，即状态集合、动作集合、状态转移函数和奖励函数。如果这个四元组中所有元素均已知，且状态集合和动作集合在有限步数内是有限集，则智能体可以对真实环境进行建模，构建一个虚拟世界来模拟真实环境中的状态和交互反应。
具体来说，当智能体知道状态转移函数 $P(s_{t+1}|s_t,a_t)$ 和奖励函数 $R(s_t,a_t)$ 后，它就能知道在某一状态下执行某一动作后能带来的奖励和环境的下一状态，这样智能体就不需要在真实环境中采取动作，直接在虚拟世界中学习和规划策略即可。这种学习方法称为**有模型强化学习**。
<div align=center>
<img width="550" src="https://github.com/Collin-Balanis/easy-rl/blob/master/docs/chapter1/img/1.35.png"/>
</div>
<div align=center>图 1.19 有模型强化学习流程</div>

> Q：有模型强化学习和免模型强化学习有什么区别？

> A：针对是否需要对真实环境建模，强化学习可以分为有模型强化学习和免模型强化学习。有模型强化学习是指根据环境中的经验，构建一个虚拟世界，同时在真实环境和虚拟世界中学习；免模
> 型强化学习是指不对环境进行建模，直接与真实环境进行交互来学习到最优策略。

> 总之，有模型强化学习相比免模型强化学习仅仅多出一个步骤，即对真实环境进行建模。因此，一些有模型的强化学习方法，也可以在免模型的强化学习方法中使用。在实际应用中，如果不清
> 楚该用有模型强化学习还是免模型强化学习，可以先思考在智能体执行动作前，是否能对下一步的状态和奖励进行预测，如果能，就能够对环境进行建模，从而采用有模型学习。

把价值函数、策略和模型几个要素放在一张图里，可以将强化学习智能体进行分类。
<div align=center>
<img width="600" src="https://github.com/Collin-Balanis/easy-rl/blob/master/docs/chapter1/img/1.36.png">
</div>
<div align=center>图1.20 强化学习智能体分类</div>

## 1.6 探索和利用

在强化学习里面，探索和利用是两个很核心的问题。 探索即我们去探索环境，通过尝试不同的动作来得到最佳的策略（带来最大奖励的策略）。 利用即我们不去尝试新的动作，而是采取已知的可以带来很大奖励的动作。

## 1.7 RL实验

按照教程顺序依次执行了程序。 增加了对RL的兴趣和认识。
