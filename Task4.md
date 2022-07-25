# Chapter6 Q-learning-State Value Function

## 1 Keywords

- **DQN(Deep Q-Network)：**  基于深度学习的Q-learninyang算法，其结合了 Value Function Approximation（价值函数近似）与神经网络技术，并采用了目标网络（Target Network）和经验回放（Experience Replay）等方法进行网络的训练。
- **State-value Function：** 本质是一种critic。其输入为actor某一时刻的state，对应的输出为一个标量，即当actor在对应的state时，预期的到过程结束时间段中获得的value的数值。
- **State-value Function Bellman Equation：** 基于state-value function的Bellman Equation，它表示在状态 $s_t$ 下带来的累积奖励 $G_t$ 的期望。
- **Q-function:** 其也被称为state-action value function。其input 是一个 state 跟 action 的 pair，即在某一个 state 采取某一个action，假设我们都使用 actor $\pi$ ，得到的 accumulated reward 的期望值有多大。
- **Target Network：** 为了解决在基于TD的Network的问题时，优化目标 $\mathrm{Q}^{\pi}\left(s_{t}, a_{t}\right) 
  =r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 左右两侧会同时变化使得训练过程不稳定，从而增大regression的难度。target network选择将上式的右部分即 $r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 固定，通过改变上式左部分的network的参数，进行regression，这也是一个DQN中比较重要的tip。
- **Exploration：**  在我们使用Q-function的时候，我们的policy完全取决于Q-function，有可能导致出现对应的action是固定的某几个数值的情况，而不像policy gradient中的output为随机的，我们再从随机的distribution中sample选择action。这样会导致我们继续训练的input的值一样，从而“加重”output的固定性，导致整个模型的表达能力的急剧下降，这也就是`探索-利用窘境难题(Exploration-Exploitation dilemma)`。所以我们使用`Epsilon Greedy`和 `Boltzmann Exploration`等Exploration方法进行优化。
- **Experience Replay（经验回放）：**  其会构建一个Replay Buffer（Replay Memory），用来保存许多data，每一个data的形式如下：在某一个 state $s_t$，采取某一个action $a_t$，得到了 reward $r_t$，然后跳到 state $s_{t+1}$。我们使用 $\pi$ 去跟环境互动很多次，把收集到的数据都放到这个 replay buffer 中。当我们的buffer”装满“后，就会自动删去最早进入buffer的data。在训练时，对于每一轮迭代都有相对应的batch（与我们训练普通的Network一样通过sample得到），然后用这个batch中的data去update我们的Q-function。综上，Q-function再sample和训练的时候，会用到过去的经验数据，所以这里称这个方法为Experience Replay，其也是DQN中比较重要的tip。

## 2 Questions

- 为什么在DQN中采用价值函数近似（Value Function Approximation）的表示方法？

  答：首先DQN为基于深度学习的Q-learning算法，而在Q-learning中，我们使用表格来存储每一个state下action的reward，即我们前面所讲的状态-动作值函数 $Q(s,a)$ 。但是在我们的实际任务中，状态量通常数量巨大并且在连续的任务中，会遇到维度灾难的问题，所以使用真正的Value Function通常是不切实际的，所以使用了价值函数近似（Value Function Approximation）的表示方法。

- critic output通常与哪几个值直接相关？

  答：critic output与state和actor有关。我们在讨论output时通常是对于一个actor下来衡量一个state的好坏，也就是state value本质上来说是依赖于actor。不同的actor在相同的state下也会有不同的output。

- 我们通常怎么衡量state value function  $V^{\pi}(s)$ ?分别的优势和劣势有哪些？

  答：

  - **基于Monte-Carlo（MC）的方法** ：本质上就是让actor与environment做互动。critic根据”统计“的结果，将actor和state对应起来，即当actor如果看到某一state $s_a$ ，将预测接下来的accumulated reward有多大如果它看到 state $s_b$，接下来accumulated reward 会有多大。 但是因为其普适性不好，其需要把所有的state都匹配到，如果我们我们是做一个简单的贪吃蛇游戏等state有限的问题，还可以进行。但是如果我们做的是一个图片型的任务，我们几乎不可能将所有的state（对应每一帧的图像）的都”记录“下来。总之，其不能对于未出现过的input state进行对应的value的输出。
  - **基于MC的Network方法：** 为了解决上面描述的Monte-Carlo（MC）方法的不足，我们将其中的state value function  $V^{\pi}(s)$ 定义为一个Network，其可以对于从未出现过的input state，根据network的泛化和拟合能力，也可以”估测“出一个value output。
  - **基于Temporal-difference（时序差分）的Network方法，即TD based Network：** 与我们再前4章介绍的MC与TD的区别一样，这里两者的区别也相同。在 MC based 的方法中，每次我们都要算 accumulated reward，也就是从某一个 state $s_a$ 一直玩到游戏结束的时候，得到的所有 reward 的总和。所以要应用 MC based 方法时，我们必须至少把这个游戏玩到结束。但有些游戏非常的长，你要玩到游戏结束才能够 update network，花的时间太长了。因此我们会采用 TD based 的方法。TD based 的方法不需要把游戏玩到底，只要在游戏的某一个情况，某一个 state $s_t$ 的时候，采取 action $a_t$ 得到 reward $r_t$ ，跳到 state $s_{t+1}$，就可以应用 TD 的方法。公式与之前介绍的TD方法类似，即：$V^{\pi}\left(s_{t}\right)=V^{\pi}\left(s_{t+1}\right)+r_{t}$。
  - **基于MC和基于TD的区别在于：** MC本身具有很大的随机性，我们可以将其 $G_a$  堪称一个random的变量，所以其最终的variance很大。而对于TD，其具有随机性的变量为 $r$ ,因为计算 $s_t$ 我们采取同一个 action，你得到的 reward 也不一定是一样的，所以对于TD来说，$r$ 是一个 random 变量。但是相对于MC的 $G_a$  的随机程度来说， $r$ 的随机性非常小，这是因为本身 $G_a$ 就是由很多的 $r$ 组合而成的。但另一个角度来说， 在TD中，我们的前提是 $r_t=V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_{t}\right)$ ,但是我们通常无法保证 $V^{\pi}\left(s_{t+1}\right)、V^{\pi}\left(s_{t}\right)$ 计算的误差为零。所以当 $V^{\pi}\left(s_{t+1}\right)、V^{\pi}\left(s_{t}\right)$  计算的不准确的话，那应用上式得到的结果，其实也会是不准的。所以 MC 跟 TD各有优劣。
  - **目前， TD 的方法是比较常见的，MC 的方法其实是比较少用的。**

- 基于我们上面说的network（基于MC）的方法，我们怎么训练这个网络呢？或者我们应该将其看做ML中什么类型的问题呢？

  答：理想状态，我们期望对于一个input state输出其无误差的reward value。也就是说这个 value function 来说，如果 input 是 state $s_a$，正确的 output 应该是$G_a$。如果 input state $s_b$，正确的output 应该是value $G_b$。所以在训练的时候，其就是一个典型的ML中的回归问题（regression problem）。所以我们实际中需要输出的仅仅是一个非精确值，即你希望在 input $s_a$ 的时候，output value 跟 $G_a$ 越近越好，input $s_b$ 的时候，output value 跟 $G_b$ 越近越好。其训练方法，和我们在训练CNN、DNN时的方法类似，就不再一一赘述。

- 基于上面介绍的基于TD的network方法，具体地，我们应该怎么训练模型呢？

  答：核心的函数为 $V^{\pi}\left(s_{t}\right)=V^{\pi}\left(s_{t+1}\right)+r_{t}$。我们将state $s_t$  作为input输入network 里，因为 $s_t$ 丢到 network 里面会得到output $V^{\pi}(s_t)$，同样将 $s_{t+1}$ 作为input输入 network 里面会得到$V^{\pi}(s_{t+1})$。同时核心函数：$V^{\pi}\left(s_{t}\right)=V^{\pi}\left(s_{t+1}\right)+r_{t}$  告诉我们，  $V^{\pi}(s_t)$ 减 $V^{\pi}(s_{t+1})$ 的值应该是 $r_t$。然后希望它们两个相减的 loss 跟 $r_t$ 尽可能地接近。这也就是我们这个network的优化目标或者说loss function。

- state-action value function（Q-function）和 state value function的有什么区别和联系？

  答：

  - state value function 的 input 是一个 state，它是根据 state 去计算出，看到这个state 以后的 expected accumulated reward 是多少。
  - state-action value function 的 input 是一个 state 跟 action 的 pair，即在某一个 state 采取某一个action，假设我们都使用 actor $\pi$ ，得到的 accumulated reward 的期望值有多大。

- Q-function的两种表示方法？

  答：

  - 当input 是 state和action的pair时，output 就是一个 scalar。
  - 当input 仅是一个 state时，output 就是好几个 value。

- 当我们有了Q-function后，我们怎么找到更好的策略 $\pi'$ 呢？或者说这个 $\pi'$ 本质来说是什么？

  答：首先， $\pi'$ 是由 $\pi^{\prime}(s)=\arg \max _{a} Q^{\pi}(s, a)$ 计算而得，其表示假设你已经 learn 出 $\pi$ 的Q-function，今天在某一个 state s，把所有可能的 action a 都一一带入这个 Q-function，看看说那一个 a 可以让 Q-function 的 value 最大，那这一个 action，就是 $\pi'$ 会采取的 action。所以根据上式决定的actoin的步骤一定比原来的 $\pi$ 要好，即$V^{\pi^{\prime}}(s) \geq V^{\pi}(s)$。

- 解决`探索-利用窘境(Exploration-Exploitation dilemma)`问题的Exploration的方法有哪些？他们具体的方法是怎样的？

  答：

  1. **Epsilon Greedy：** 我们有$1-\varepsilon$ 的机率，通常 $\varepsilon$ 很小，完全按照Q-function 来决定action。但是有 $\varepsilon$ 的机率是随机的。通常在实现上 $\varepsilon$ 会随着时间递减。也就是在最开始的时候。因为还不知道那个action 是比较好的，所以你会花比较大的力气在做 exploration。接下来随着training 的次数越来越多。已经比较确定说哪一个Q 是比较好的。你就会减少你的exploration，你会把 $\varepsilon$ 的值变小，主要根据Q-function 来决定你的action，比较少做random，这是**Epsilon Greedy**。
  2. **Boltzmann Exploration：** 这个方法就比较像是 policy gradient。在 policy gradient 里面network 的output 是一个 expected action space 上面的一个的 probability distribution。再根据 probability distribution 去做 sample。所以也可以根据Q value 去定一个 probability distribution，假设某一个 action 的 Q value 越大，代表它越好，我们采取这个 action 的机率就越高。这是**Boltzmann Exploration**。

- 我们使用Experience Replay（经验回放）有什么好处？

  答：

  1. 首先，在强化学习的整个过程中， 最花时间的 step 是在跟环境做互动，使用GPU乃至TPU加速来训练 network 相对来说是比较快的。而用 replay buffer 可以减少跟环境做互动的次数，因为在训练的时候，我们的 experience 不需要通通来自于某一个policy（或者当前时刻的policy）。一些过去的 policy 所得到的 experience 可以放在 buffer 里面被使用很多次，被反复的再利用，这样让你的 sample 到 experience 的利用是高效的。
  2. 另外，在训练网络的时候，其实我们希望一个 batch 里面的 data 越 diverse 越好。如果你的 batch 里面的 data 都是同样性质的，我们的训练出的模型拟合能力不会很乐观。如果 batch 里面都是一样的 data，你 train 的时候，performance 会比较差。我们希望 batch data 越 diverse 越好。那如果 buffer 里面的那些 experience 通通来自于不同的 policy ，那你 sample 到的一个 batch 里面的 data 会是比较 diverse 。这样可以保证我们模型的性能至少不会很差。

- 在Experience Replay中我们是要观察 $\pi$ 的 value，里面混杂了一些不是 $\pi$ 的 experience ，这会有影响吗？

  答：没关系。这并不是因为过去的 $\pi$ 跟现在的 $\pi$ 很像， 就算过去的$\pi$ 没有很像，其实也是没有关系的。主要的原因是我们并不是去sample 一个trajectory，我们只sample 了一个experience，所以跟是不是 off-policy 这件事是没有关系的。就算是off-policy，就算是这些 experience 不是来自于 $\pi$，我们其实还是可以拿这些 experience 来估测 $Q^{\pi}(s,a)$。

- DQN（Deep Q-learning）和Q-learning有什么异同点？

  答：整体来说，从名称就可以看出，两者的目标价值以及价值的update方式基本相同，另外一方面，不同点在于：

  - 首先，DQN 将 Q-learning 与深度学习结合，用深度网络来近似动作价值函数，而 Q-learning 则是采用表格存储。
  - DQN 采用了我们前面所描述的经验回放（Experience Replay）训练方法，从历史数据中随机采样，而 Q-learning 直接采用下一个状态的数据进行学习。


## 3 Something About Interview

- 高冷的面试官：请问DQN（Deep Q-Network）是什么？其两个关键性的技巧分别是什么？

  答：Deep Q-Network是基于深度学习的Q-learning算法，其结合了 Value Function Approximation（价值函数近似）与神经网络技术，并采用了目标网络（Target Network）和经验回放（Experience Replay）的方法进行网络的训练。

- 高冷的面试官：接上题，DQN中的两个trick：目标网络和experience replay的具体作用是什么呢？

  答：在DQN中某个动作值函数的更新依赖于其他动作值函数。如果我们一直更新值网络的参数，会导致
  更新目标不断变化，也就是我们在追逐一个不断变化的目标，这样势必会不太稳定。 为了解决在基于TD的Network的问题时，优化目标 $\mathrm{Q}^{\pi}\left(s_{t}, a_{t}\right) =r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 左右两侧会同时变化使得训练过程不稳定，从而增大regression的难度。target network选择将上式的右部分即 $r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 固定，通过改变上式左部分的network的参数，进行regression。对于经验回放，其会构建一个Replay Buffer（Replay Memory），用来保存许多data，每一个data的形式如下：在某一个 state $s_t$，采取某一个action $a_t$，得到了 reward $r_t$，然后跳到 state $s_{t+1}$。我们使用 $\pi$ 去跟环境互动很多次，把收集到的数据都放到这个 replay buffer 中。当我们的buffer”装满“后，就会自动删去最早进入buffer的data。在训练时，对于每一轮迭代都有相对应的batch（与我们训练普通的Network一样通过sample得到），然后用这个batch中的data去update我们的Q-function。也就是，Q-function再sample和训练的时候，会用到过去的经验数据，也可以消除样本之间的相关性。

- 高冷的面试官：DQN（Deep Q-learning）和Q-learning有什么异同点？

  答：整体来说，从名称就可以看出，两者的目标价值以及价值的update方式基本相同，另外一方面，不同点在于：

  - 首先，DQN 将 Q-learning 与深度学习结合，用深度网络来近似动作价值函数，而 Q-learning 则是采用表格存储。
  - DQN 采用了我们前面所描述的经验回放（Experience Replay）训练方法，从历史数据中随机采样，而 Q-learning 直接采用下一个状态的数据进行学习。

- 高冷的面试官：请问，随机性策略和确定性策略有什么区别吗？

  答：随机策略表示为某个状态下动作取值的分布，确定性策略在每个状态只有一个确定的动作可以选。
  从熵的角度来说，确定性策略的熵为0，没有任何随机性。随机策略有利于我们进行适度的探索，确定
  性策略的探索问题更为严峻。

- 高冷的面试官：请问不打破数据相关性，神经网络的训练效果为什么就不好？

  答：在神经网络中通常使用随机梯度下降法。随机的意思是我们随机选择一些样本来增量式的估计梯度，比如常用的
  采用batch训练。如果样本是相关的，那就意味着前后两个batch的很可能也是相关的，那么估计的梯度也会呈现
  出某种相关性。如果不幸的情况下，后面的梯度估计可能会抵消掉前面的梯度量。从而使得训练难以收敛。


# Chapter7 Q-learning-Double DQN

## 1 Keywords

- **Double DQN：** 在Double DQN中存在有两个 Q-network，首先，第一个 Q-network，决定的是哪一个 action 的 Q value 最大，从而决定了你的action。另一方面， Q value 是用 $Q'$ 算出来的，这样就可以避免 over estimate 的问题。具体来说，假设我们有两个 Q-function，假设第一个Q-function 它高估了它现在选出来的action a，那没关系，只要第二个Q-function $Q'$ 没有高估这个action a 的值，那你算出来的，就还是正常的值。
- **Dueling DQN：** 将原来的DQN的计算过程分为**两个path**。对于第一个path，会计算一个于input state有关的一个标量 $V(s)$；对于第二个path，会计算出一个vector $A(s,a)$ ，其对应每一个action。最后的网络是将两个path的结果相加，得到我们最终需要的Q value。用一个公式表示也就是 $Q(s,a)=V(s)+A(s,a)$ 。 
- **Prioritized Experience Replay （优先经验回放）：** 这个方法是为了解决我们在chapter6中提出的**Experience Replay（经验回放）**方法不足进一步优化提出的。我们在使用Experience Replay时是uniformly取出的experience buffer中的sample data，这里并没有考虑数据间的权重大小。例如，我们应该将那些train的效果不好的data对应的权重加大，即其应该有更大的概率被sample到。综上， prioritized experience replay 不仅改变了 sample data 的 distribution，还改变了 training process。
- **Noisy Net：** 其在每一个episode 开始的时候，即要和环境互动的时候，将原来的Q-function 的每一个参数上面加上一个Gaussian noise。那你就把原来的Q-function 变成$\tilde{Q}$ ，即**Noisy Q-function**。同样的我们把每一个network的权重等参数都加上一个Gaussian noise，就得到一个新的network $\tilde{Q}$。我们会使用这个新的network从与环境互动开始到互动结束。
- **Distributional Q-function：** 对于DQN进行model distribution。将最终的网络的output的每一类别的action再进行distribution。
- **Rainbow：** 也就是将我们这两节内容所有的七个tips综合起来的方法，7个方法分别包括：DQN、DDQN、Prioritized DDQN、Dueling DDQN、A3C、Distributional DQN、Noisy DQN，进而考察每一个方法的贡献度或者是否对于与环境的交互式正反馈的。

## 2 Questions

- 为什么传统的DQN的效果并不好？参考公式 $Q(s_t ,a_t)=r_t+\max_{a}Q(s_{t+1},a)$ 

  答：因为实际上在做的时候，是要让左边这个式子跟右边这个 target 越接近越好。比较容易可以发现target 的值很容易一不小心就被设得太高。因为在算这个 target 的时候，我们实际上在做的事情是看哪一个a 可以得到最大的Q value，就把它加上去，就变成我们的target。

  举例来说，现在有 4 个 actions，本来其实它们得到的值都是差不多的，它们得到的reward 都是差不多的。但是在estimate 的时候，那毕竟是个network。所以estimate 的时候是有误差的。所以假设今天是第一个action它被高估了，假设绿色的东西代表是被高估的量，它被高估了，那这个target 就会选这个action。然后就会选这个高估的Q value来加上$r_t$，来当作你的target。如果第4 个action 被高估了，那就会选第4 个action 来加上$r_t$ 来当作你的target value。所以你总是会选那个Q value 被高估的，你总是会选那个reward 被高估的action 当作这个max 的结果去加上$r_t$ 当作你的target。所以你的target 总是太大。

- 接着上个思考题，我们应该怎么解决target 总是太大的问题呢？

  答： 我们可以使用Double DQN解决这个问题。首先，在 Double DQN 里面，选 action 的 Q-function 跟算 value 的 Q-function不同。在原来的DQN 里面，你穷举所有的 a，把每一个a 都带进去， 看哪一个 a 可以给你的 Q value 最高，那你就把那个 Q value 加上$r_t$。但是在 Double DQN 里面，你**有两个 Q-network**，第一个 Q-network，决定哪一个 action 的 Q value 最大，你用第一个 Q-network 去带入所有的 a，去看看哪一个Q value 最大。然后你决定你的action 以后，你的 Q value 是用 $Q'$ 算出来的，这样子有什么好处呢？为什么这样就可以避免 over estimate 的问题呢？因为今天假设我们有两个 Q-function，假设第一个Q-function 它高估了它现在选出来的action a，那没关系，只要第二个Q-function $Q'$ 没有高估这个action a 的值，那你算出来的，就还是正常的值。假设反过来是 $Q'$ 高估了某一个action 的值，那也没差， 因为反正只要前面这个Q 不要选那个action 出来就没事了。

- 哪来 Q  跟 $Q'$ 呢？哪来两个 network 呢？

  答：在实现上，你有两个 Q-network， 一个是 target 的 Q-network，一个是真正你会 update 的 Q-network。所以在 Double DQN 里面，你的实现方法会是拿你会 update 参数的那个 Q-network 去选action，然后你拿target 的network，那个固定住不动的network 去算value。而 Double DQN 相较于原来的 DQN 的更改是最少的，它几乎没有增加任何的运算量，连新的network 都不用，因为你原来就有两个network 了。你唯一要做的事情只有，本来你在找最大的a 的时候，你在决定这个a 要放哪一个的时候，你是用$Q'$ 来算，你是用target network 来算，现在改成用另外一个会 update 的 Q-network 来算。

- 如何理解Dueling DQN的模型变化带来的好处？

  答：对于我们的 $Q(s,a)$ 其对应的state由于为table的形式，所以是离散的，而实际中的state不是离散的。对于 $Q(s,a)$ 的计算公式， $Q(s,a)=V(s)+A(s,a)$ 。其中的 $V(s)$ 是对于不同的state都有值，对于 $A(s,a)$ 对于不同的state都有不同的action对应的值。所以本质上来说，我们最终的矩阵 $Q(s,a)$ 的结果是将每一个 $V(s)$ 加到矩阵 $A(s,a)$ 中得到的。从模型的角度考虑，我们的network直接改变的 $Q(s,a)$ 而是 更改的 $V、A$ 。但是有时我们update时不一定会将 $V(s)$ 和 $Q(s,a)$ 都更新。我们将其分成两个path后，我们就不需要将所有的state-action pair都sample一遍，我们可以使用更高效的estimate Q value方法将最终的 $Q(s,a)$ 计算出来。

- 使用MC和TD平衡方法的优劣分别有哪些？

  答：

  - 优势：因为我们现在 sample 了比较多的step，之前是只sample 了一个step， 所以某一个step 得到的data 是真实值，接下来都是Q value 估测出来的。现在sample 比较多step，sample N 个step 才估测value，所以估测的部分所造成的影响就会比小。
  - 劣势：因为我们的 reward 比较多，当我们把 N 步的 reward 加起来，对应的 variance 就会比较大。但是我们可以选择通过调整 N 值，去在variance 跟不精确的 Q 之间取得一个平衡。这里介绍的参数 N 就是一个hyper parameter，你要调这个N 到底是多少，你是要多 sample 三步，还是多 sample 五步。



## 3 Something About Interview

- 高冷的面试官：DQN都有哪些变种？引入状态奖励的是哪种？

  答：DQN三个经典的变种：Double DQN、Dueling DQN、Prioritized Replay Buffer。

  - Double-DQN：将动作选择和价值估计分开，避免价值过高估计。
  - Dueling-DQN：将Q值分解为状态价值和优势函数，得到更多有用信息。
  - Prioritized Replay Buffer：将经验池中的经验按照优先级进行采样。

- 简述double DQN原理？

  答：DQN由于总是选择当前值函数最大的动作值函数来更新当前的动作值函数，因此存在着过估计问题（估计的值函数大于真实的值函数）。为了解耦这两个过程，double DQN 使用了两个值网络，一个网络用来执行动作选择，然后用另一个值函数对一个的动作值更新当前网络。

- 高冷的面试官：请问Dueling DQN模型有什么优势呢？

  答：对于我们的 $Q(s,a)$ 其对应的state由于为table的形式，所以是离散的，而实际中的state不是离散的。对于 $Q(s,a)$ 的计算公式， $Q(s,a)=V(s)+A(s,a)$ 。其中的 $V(s)$ 是对于不同的state都有值，对于 $A(s,a)$ 对于不同的state都有不同的action对应的值。所以本质上来说，我们最终的矩阵 $Q(s,a)$ 的结果是将每一个 $V(s)$ 加到矩阵 $A(s,a)$ 中得到的。从模型的角度考虑，我们的network直接改变的 $Q(s,a)$ 而是更改的 $V、A$ 。但是有时我们update时不一定会将 $V(s)$ 和 $Q(s,a)$ 都更新。我们将其分成两个path后，我们就不需要将所有的state-action pair都sample一遍，我们可以使用更高效的estimate Q value方法将最终的 $Q(s,a)$ 计算出来。



# Chapter8 Q-learning for Continuous Actions

## Questions

- Q-learning相比于policy gradient based方法为什么训练起来效果更好，更平稳？

  答：在 Q-learning 中，只要能够 estimate 出Q-function，就可以保证找到一个比较好的 policy，同样的只要能够 estimate 出 Q-function，就保证可以 improve 对应的 policy。而因为 estimate Q-function 作为一个回归问题，是比较容易的。在这个回归问题中， 我们可以时刻观察我们的模型训练的效果是不是越来越好，一般情况下我们只需要关注 regression 的 loss 有没有下降，你就知道你的 model learn 的好不好。所以 estimate Q-function 相较于 learn 一个 policy 是比较容易的。你只要 estimate Q-function，就可以保证说现在一定会得到比较好的 policy，同样其也比较容易操作。

- Q-learning在处理continuous action时存在什么样的问题呢？

  答：在日常的问题中，我们的问题都是continuous action的，例如我们的 agent 要做的事情是开自驾车，它要决定说它方向盘要左转几度， 右转几度，这就是 continuous 的；假设我们的 agent 是一个机器人，假设它身上有 50 个关节，它的每一个 action 就对应到它身上的这 50 个关节的角度，而那些角度也是 continuous 的。

  然而在解决Q-learning问题时，很重要的一步是要求能够解对应的优化问题。当我们 estimate 出Q-function $Q(s,a)$ 以后,必须要找到一个 action，它可以让 $Q(s,a)$ 最大。假设 action 是 discrete 的，那 a 的可能性都是有限的。但如果action是continuous的情况下，我们就不能像离散的action一样，穷举所有可能的continuous action了。

  为了解决这个问题，有以下几种solutions：

  - 第一个解决方法：我们可以使用所谓的sample方法，即随机sample出N个可能的action，然后一个一个带到我们的Q-function中，计算对应的N个Q value比较哪一个的值最大。但是这个方法因为是sample所以不会非常的精确。
  - 第二个解决方法：我们将这个continuous action问题，看为一个优化问题，从而自然而然地想到了可以用gradient ascend去最大化我们的目标函数。具体地，我们将action看为我们的变量，使用gradient ascend方法去update action对应的Q-value。但是这个方法通常的时间花销比较大，因为是需要迭代运算的。
  - 第三个解决方法：设计一个特别的network架构，设计一个特别的Q-function，使得解我们 argmax Q-value的问题变得非常容易。也就是这边的 Q-function 不是一个 general 的 Q-function，特别设计一下它的样子，让你要找让这个 Q-function 最大的 a 的时候非常容易。但是这个方法的function不能随意乱设，其必须有一些额外的限制。具体的设计方法，可以我们的chapter8的详细教程。
  - 第四个解决方法：不用Q-learning，毕竟用其处理continuous的action比较麻烦。


# Chapter9 Actor-Critic

## 1 Keywords

- **A2C：** Advantage Actor-Critic的缩写，一种Actor-Critic方法。

- **A3C：** Asynchronous（异步的）Advantage Actor-Critic的缩写，一种改进的Actor-Critic方法，通过异步的操作，进行RL模型训练的加速。
-  **Pathwise Derivative Policy Gradient：** 其为使用 Q-learning 解 continuous action 的方法，也是一种 Actor-Critic 方法。其会对于actor提供value最大的action，而不仅仅是提供某一个action的好坏程度。

## 2 Questions

- 整个Advantage actor-critic（A2C）算法的工作流程是怎样的？

  答：在传统的方法中，我们有一个policy $\pi$ 以及一个初始的actor与environment去做互动，收集数据以及反馈。通过这些每一步得到的数据与反馈，我们就要进一步更新我们的policy $\pi$ ，通常我们所使用的方式是policy gradient。但是对于actor-critic方法，我们不是直接使用每一步得到的数据和反馈进行policy $\pi$ 的更新，而是使用这些数据进行 estimate value function，这里我们通常使用的算法包括前几个chapters重点介绍的TD和MC等算法以及他们的优化算法。接下来我们再基于value function来更新我们的policy，公式如下：
  $$
  \nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}}\left(r_{t}^{n}+V^{\pi}\left(s_{t+1}^{n}\right)-V^{\pi}\left(s_{t}^{n}\right)\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)
  $$
  其中，上式中的 $r_{t}^{n}+V^{\pi}\left(s_{t+1}^{n}\right)-V^{\pi}\left(s_{t}^{n}\right)$ 我们称为Advantage function，我们通过上式得到新的policy后，再去与environment进行交互，然后再重复我们的estimate value function的操作，再用value function来更新我们的policy。以上的整个方法我们称为Advantage Actor-Critic。

- 在实现 Actor-Critic 的时候，有哪些我们用到的tips?

  答：与我们上一章讲述的东西有关：

  1. **estimate 两个 network：** 一个是estimate V function，另外一个是 policy 的 network，也就是你的 actor。 V-network的input 是一个 state，output 是一个 scalar。然后 actor 这个 network的input 是一个 state，output 是一个 action 的 distribution。这两个 network，actor 和 critic 的 input 都是 s，所以它们前面几个 layer，其实是可以 share 的。尤其是假设你今天是玩 Atari 游戏，input 都是 image。那 input 那个 image 都非常复杂，image 很大，通常前面都会用一些 CNN 来处理，把那些 image 抽象成 high level 的 information，所以对 actor 跟 critic 来说是可以共用的。我们可以让 actor 跟 critic 的前面几个 layer 共用同一组参数。那这一组参数可能是 CNN。先把 input 的 pixel 变成比较 high level 的信息，然后再给 actor 去决定说它要采取什么样的行为，给这个 critic，给 value function 去计算 expected reward。
  2. **exploration 机制：** 其目的是对policy $\pi$ 的 output 的分布进行一个限制，从而使得 distribution 的 entropy 不要太小，即希望不同的 action 被采用的机率平均一点。这样在 testing 的时候，它才会多尝试各种不同的 action，才会把这个环境探索的比较好，才会得到比较好的结果。

- A3C（Asynchronous Advantage Actor-Critic）在训练是回有很多的worker进行异步的工作，最后再讲他们所获得的“结果”再集合到一起。那么其具体的如何运作的呢？

  答：A3C一开始会有一个 global network。它们有包含 policy 的部分和 value 的部分，假设它的参数就是 $\theta_1$。对于每一个 worker 都用一张 CPU 训练（举例子说明），第一个 worker 就把 global network 的参数 copy 过来，每一个 worker 工作前都会global network 的参数 copy 过来。然后这个worker就要去跟environment进行交互，每一个 actor 去跟environment做互动后，就会计算出 gradient并且更新global network的参数。这里要注意的是，所有的 actor 都是平行跑的、之间没有交叉。所以每个worker都是在global network“要”了一个参数以后，做完就把参数传回去。所以当第一个 worker 做完想要把参数传回去的时候，本来它要的参数是 $\theta_1$，等它要把 gradient 传回去的时候。可能别人已经把原来的参数覆盖掉，变成 $\theta_2$了。但是没有关系，它一样会把这个 gradient 就覆盖过去就是了。

- 对比经典的Q-learning算法，我们的Pathwise Derivative Policy Gradient有哪些改进之处？

  答：

  1. 首先，把 $Q(s,a)$ 换成 了 $\pi$，之前是用 $Q(s,a)$ 来决定在 state $s_t$ 产生那一个 action, $a_{t}$ 现在是直接用 $\pi$ 。原先我们需要解 argmax 的问题，现在我们直接训练了一个 actor。这个 actor input $s_t$ 就会告诉我们应该采取哪一个 $a_{t}$。综上，本来 input $s_t$，采取哪一个 $a_t$，是 $Q(s,a)$ 决定的。在 Pathwise Derivative Policy Gradient 里面，我们会直接用 $\pi$ 来决定。
  2. 另外，原本是要计算在 $s_{i+1}$ 时对应的 policy 采取的 action a 会得到多少的 Q value，那你会采取让 $\hat{Q}$ 最大的那个 action a。现在因为我们不需要再解argmax 的问题。所以现在我们就直接把 $s_{i+1}$ 代入到 policy $\pi$ 里面，直接就会得到在 $s_{i+1}$ 下，哪一个 action 会给我们最大的 Q value，那你在这边就会 take 那一个 action。在 Q-function 里面，有两个 Q network，一个是真正的 Q network，另外一个是 target Q network。那实际上你在 implement 这个 algorithm 的时候，你也会有两个 actor，你会有一个真正要 learn 的 actor $\pi$，你会有一个 target actor $\hat{\pi}$ 。但现在因为哪一个 action a 可以让 $\hat{Q}$ 最大这件事情已经被用那个 policy 取代掉了，所以我们要知道哪一个 action a 可以让 $\hat{Q}$ 最大，就直接把那个 state 带到 $\hat{\pi}$ 里面，看它得到哪一个 a，就用那一个 a，其也就是会让 $\hat{Q}(s,a)$ 的值最大的那个 a 。
  3. 还有，之前只要 learn Q，现在你多 learn 一个 $\pi$，其目的在于maximize Q-function，希望你得到的这个 actor，它可以让你的 Q-function output 越大越好，这个跟 learn GAN 里面的 generator 的概念类似。
  4. 最后，与原来的 Q-function 一样。我们要把 target 的 Q-network 取代掉，你现在也要把 target policy 取代掉。


## 3 Something About Interview

- 高冷的面试官：请简述一下A3C算法吧，另外A3C是on-policy还是off-policy呀？

  答：A3C就是异步优势演员-评论家方法（Asynchronous Advantage Actor-Critic）：评论家学习值函数，同时有多个actor并行训练并且不时与全局参数同步。A3C旨在用于并行训练，是 on-policy 的方法。 

- 高冷的面试官：请问Actor - Critic有何优点呢？

  答：

  - 相比以值函数为中心的算法，Actor - Critic应用了策略梯度的做法，这能让它在连续动作或者高维动作空间中选取合适的动作，而 Q-learning 做这件事会很困难甚至瘫痪。
  - 相比单纯策略梯度，Actor - Critic应用了Q-learning或其他策略评估的做法，使得Actor Critic能进行单步更新而不是回合更新，比单纯的Policy Gradient的效率要高。

- 高冷的面试官：请问A3C算法具体是如何异步更新的？

  答：下面是算法大纲：

  - 定义全局参数 $\theta$ 和 $w$ 以及特定线程参数 $θ′$ 和 $w′$。
  - 初始化时间步 $t=1$。
  - 当 $T<=T_{max}$：
    - 重置梯度：$dθ=0$ 并且 $dw=0$。
    - 将特定于线程的参数与全局参数同步：$θ′=θ$ 以及 $w′=w$。
    - 令 $t_{start} =t$ 并且随机采样一个初始状态 $s_t$。
    - 当 （$s_t!=$ 终止状态）并$t−t_{start}<=t_{max}$：
      - 根据当前线程的策略选择当前执行的动作 $a_t∼π_{θ′}(a_t|s_t)$，执行动作后接收回报$r_t$然后转移到下一个状态st+1。
      - 更新 t 以及 T：t=t+1 并且 T=T+1。
    - 初始化保存累积回报估计值的变量
    - 对于 $i=t_1,…,t_{start}$：
      - r←γr+ri；这里 r 是 Gi 的蒙特卡洛估计。
      - 累积关于参数 θ′的梯度：$dθ←dθ+∇θ′logπθ′(ai|si)(r−Vw′(si))$;
      - 累积关于参数 w′ 的梯度：$dw←dw+2(r−Vw′(si))∇w′(r−Vw′(si))$.
    - 分别使用 dθ以及 dw异步更新 θ以及 w。
    
- 高冷的面试官：Actor-Critic两者的区别是什么？

  答：Actor是策略模块，输出动作；critic是判别器，用来计算值函数。

- 高冷的面试官：actor-critic框架中的critic起了什么作用？

  答：critic表示了对于当前决策好坏的衡量。结合策略模块，当critic判别某个动作的选择时有益的，策略就更新参数以增大该动作出现的概率，反之降低动作出现的概率。

- 高冷的面试官：简述A3C的优势函数？
  
  答：$A(s,a)=Q(s,a)-V(s)$是为了解决value-based方法具有高变异性。它代表着与该状态下采取的平均行动相比所取得的进步。

    - 如果 A(s,a)>0: 梯度被推向了该方向
    - 如果 A(s,a)<0: (我们的action比该state下的平均值还差) 梯度被推向了反方

    但是这样就需要两套 value function，所以可以使用TD error 做估计：$A(s,a)=r+\gamma V(s')-V(s)$。
