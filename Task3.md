# Chapter4 梯度策略 

## 1 Keywords

- **policy（策略）：** 每一个actor中会有对应的策略，这个策略决定了actor的行为。具体来说，Policy 就是给一个外界的输入，然后它会输出 actor 现在应该要执行的行为。**一般地，我们将policy写成 $\pi$ 。**
- **Return（回报）：** 一个回合（Episode）或者试验（Trial）所得到的所有的reward的总和，也被人们称为Total reward。**一般地，我们用 $R$ 来表示它。**
- **Trajectory：** 一个试验中我们将environment 输出的 $s$ 跟 actor 输出的行为 $a$，把这个 $s$ 跟 $a$ 全部串起来形成的集合，我们称为Trajectory，即  $\text { Trajectory } \tau=\left\{s_{1}, a_{1}, s_{2}, a_{2}, \cdots, s_{t}, a_{t}\right\}$。
- **Reward function：** 根据在某一个 state 采取的某一个 action 决定说现在这个行为可以得到多少的分数，它是一个 function。也就是给一个 $s_1$，$a_1$，它告诉你得到 $r_1$。给它 $s_2$ ，$a_2$，它告诉你得到 $r_2$。 把所有的 $r$ 都加起来，我们就得到了 $R(\tau)$ ，代表某一个 trajectory $\tau$ 的 reward。
- **Expected reward：** $\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]$。
- **REINFORCE：** 基于策略梯度的强化学习的经典算法，其采用回合更新的模式。

## 2 Questions

- 如果我们想让机器人自己玩video game, 那么强化学习中三个组成（actor、environment、reward function）部分具体分别是什么？

  答：actor 做的事情就是去操控游戏的摇杆， 比如说向左、向右、开火等操作；environment 就是游戏的主机， 负责控制游戏的画面负责控制说，怪物要怎么移动， 你现在要看到什么画面等等；reward function 就是当你做什么事情，发生什么状况的时候，你可以得到多少分数， 比如说杀一只怪兽得到 20 分等等。

- 在一个process中，一个具体的trajectory $s_1$,$a_1$, $s_2$ , $a_2$ 出现的概率取决于什么？

  答：

  1. 一部分是 **environment 的行为**， environment 的 function 它内部的参数或内部的规则长什么样子。 $p(s_{t+1}|s_t,a_t)$这一项代表的是 environment， environment 这一项通常你是无法控制它的，因为那个是人家写好的，或者已经客观存在的。

  2. 另一部分是 **agent 的行为**，你能控制的是 $p_\theta(a_t|s_t)$。给定一个 $s_t$， actor 要采取什么样的 $a_t$ 会取决于你 actor 的参数 $\theta$， 所以这部分是 actor 可以自己控制的。随着 actor 的行为不同，每个同样的 trajectory， 它就会有不同的出现的概率。

- 当我们在计算 maximize expected reward时，应该使用什么方法？

  答： **gradient ascent（梯度上升）**，因为要让它越大越好，所以是 gradient ascent。Gradient ascent 在 update 参数的时候要加。要进行 gradient ascent，我们先要计算 expected reward $\bar{R}$ 的 gradient 。我们对 $\bar{R}$ 取一个 gradient，这里面只有 $p_{\theta}(\tau)$ 是跟 $\theta$ 有关，所以 gradient 就放在 $p_{\theta}(\tau)$ 这个地方。

- 我们应该如何理解梯度策略的公式呢？

  答：
  $$
  \begin{aligned}
  E_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] &\approx \frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(\tau^{n}\right) \\
  &=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)
  \end{aligned}
  $$
   $p_{\theta}(\tau)$ 里面有两项，$p(s_{t+1}|s_t,a_t)$ 来自于 environment，$p_\theta(a_t|s_t)$ 是来自于 agent。 $p(s_{t+1}|s_t,a_t)$ 由环境决定从而与 $\theta$ 无关，因此 $\nabla \log p(s_{t+1}|s_t,a_t) =0 $。因此 $\nabla p_{\theta}(\tau)=
  \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)$。 公式的具体推导可见我们的教程。

  具体来说：

  *  假设你在 $s_t$ 执行 $a_t$，最后发现 $\tau$ 的 reward 是正的， 那你就要增加这一项的概率，即增加在 $s_t$ 执行 $a_t$ 的概率。
  *  反之，在 $s_t$ 执行 $a_t$ 会导致$\tau$  的 reward 变成负的， 你就要减少这一项的概率。

- 我们可以使用哪些方法来进行gradient ascent的计算？

  答：用 gradient ascent 来 update 参数，对于原来的参数 $\theta$ ，可以将原始的 $\theta$  加上更新的 gradient 这一项，再乘以一个 learning rate，learning rate 其实也是要调的，和神经网络一样，我们可以使用 Adam、RMSProp 等优化器对其进行调整。

- 我们进行基于梯度策略的优化时的小技巧有哪些？

  答：

  1. **Add a baseline：**为了防止所有的reward都大于0，从而导致每一个stage和action的变换，会使得每一项的概率都会上升。所以通常为了解决这个问题，我们把reward 减掉一项叫做 b，这项 b 叫做 baseline。你减掉这项 b 以后，就可以让 $R(\tau^n)-b$ 这一项， 有正有负。 所以如果得到的 total reward $R(\tau^n)$ 大于 b 的话，就让它的概率上升。如果这个 total reward 小于 b，就算它是正的，正的很小也是不好的，你就要让这一项的概率下降。 如果$R(\tau^n)<b$  ， 你就要让这个 state 采取这个 action 的分数下降 。这样也符合常理。但是使用baseline会让本来reward很大的“行为”的reward变小，降低更新速率。
  2. **Assign suitable credit：** 首先第一层，本来的 weight 是整场游戏的 reward 的总和。那现在改成从某个时间 $t$ 开始，假设这个 action 是在 t 这个时间点所执行的，从 $t$ 这个时间点，一直到游戏结束所有 reward 的总和，才真的代表这个 action 是好的还是不好的；接下来我们再进一步，我们把未来的reward做一个discount，这里我们称由此得到的reward的和为**Discounted Return(折扣回报)** 。
  3. 综合以上两种tip，我们将其统称为**Advantage function**， 用 `A` 来代表 advantage function。Advantage function 是 dependent on s and a，我们就是要计算的是在某一个 state s 采取某一个 action a 的时候，advantage function 有多大。
  4. Advantage function 的意义就是，假设我们在某一个 state $s_t$ 执行某一个 action $a_t$，相较于其他可能的 action，它有多好。它在意的不是一个绝对的好，而是相对的好，即相对优势(relative advantage)。因为会减掉一个 b，减掉一个 baseline， 所以这个东西是相对的好，不是绝对的好。 $A^{\theta}\left(s_{t}, a_{t}\right)$ 通常可以是由一个 network estimate 出来的，这个 network 叫做 critic。
  
- 对于梯度策略的两种方法，蒙特卡洛（MC）强化学习和时序差分（TD）强化学习两个方法有什么联系和区别？

  答：

  1. **两者的更新频率不同**，蒙特卡洛强化学习方法是**每一个episode更新一次**，即需要经历完整的状态序列后再更新（比如我们的贪吃蛇游戏，贪吃蛇“死了”游戏结束后再更新），而对于时序差分强化学习方法是**每一个step就更新一次** ，（比如我们的贪吃蛇游戏，贪吃蛇每移动一次（或几次）就进行更新）。相对来说，时序差分强化学习方法比蒙特卡洛强化学习方法更新的频率更快。
  2. 时序差分强化学习能够在知道一个小step后就进行学习，相比于蒙特卡洛强化学习，其更加**快速、灵活**。
  3. 具体举例来说：假如我们要优化开车去公司的通勤时间。对于此问题，每一次通勤，我们将会到达不同的路口。对于时序差分（TD）强化学习，其会对于每一个经过的路口都会计算时间，例如在路口 A 就开始更新预计到达路口 B、路口 C $\cdots \cdots$, 以及到达公司的时间；而对于蒙特卡洛（MC）强化学习，其不会每经过一个路口就更新时间，而是到达最终的目的地后，再修改每一个路口和公司对应的时间。

- 请详细描述REINFORCE的计算过程。

  答：首先我们需要根据一个确定好的policy model来输出每一个可能的action的概率，对于所有的action的概率，我们使用sample方法（或者是随机的方法）去选择一个action与环境进行交互，同时环境就会给我们反馈一整个episode数据。对于此episode数据输入到learn函数中，并根据episode数据进行loss function的构造，通过adam等优化器的优化，再来更新我们的policy model。


## 3 Something About Interview

- 高冷的面试官：同学来吧，给我手工推导一下策略梯度公式的计算过程。

  答：首先我们目的是最大化reward函数，即调整 $\theta$ ，使得期望回报最大，可以用公式表示如下
  $$
  J(\theta)=E_{\tau \sim p_{\theta(\mathcal{T})}}[\sum_tr(s_t,a_t)]
  $$
  对于上面的式子， $\tau$ 表示从从开始到结束的一条完整路径。通常，对于最大化问题，我们可以使用梯度上升算法来找到最大值，即
  $$
  \theta^* = \theta + \alpha\nabla J({\theta})
  $$
  所以我们仅仅需要计算（更新）$\nabla J({\theta})$  ，也就是计算回报函数 $J({\theta})$ 关于 $\theta$ 的梯度，也就是策略梯度，计算方法如下：
  $$\begin{aligned}
  \nabla_{\theta}J(\theta) &= \int {\nabla}_{\theta}p_{\theta}(\tau)r(\tau)d_{\tau} \\
  &= \int p_{\theta}{\nabla}_{\theta}logp_{\theta}(\tau)r(\tau)d_{\tau} \\
  &= E_{\tau \sim p_{\theta}(\tau)}[{\nabla}_{\theta}logp_{\theta}(\tau)r(\tau)]
  \end{aligned}$$
  接着我们继续讲上式展开，对于 $p_{\theta}(\tau)$ ，即 $p_{\theta}(\tau|{\theta})$ :
  $$
  p_{\theta}(\tau|{\theta}) = p(s_1)\prod_{t=1}^T \pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)
  $$
  取对数后为：
  $$
  logp_{\theta}(\tau|{\theta}) = logp(s_1)+\sum_{t=1}^T log\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)
  $$
  继续求导：
  $$
  \nabla logp_{\theta}(\tau|{\theta}) = \sum_{t=1}^T \nabla_{\theta}log \pi_{\theta}(a_t|s_t)
  $$
  带入第三个式子，可以将其化简为：
  $$\begin{aligned}
  \nabla_{\theta}J(\theta) &= E_{\tau \sim p_{\theta}(\tau)}[{\nabla}_{\theta}logp_{\theta}(\tau)r(\tau)] \\
  &= E_{\tau \sim p_{\theta}}[(\nabla_{\theta}log\pi_{\theta}(a_t|s_t))(\sum_{t=1}^Tr(s_t,a_t))] \\ 
  &= \frac{1}{N}\sum_{i=1}^N[(\sum_{t=1}^T\nabla_{\theta}log \pi_{\theta}(a_{i,t}|s_{i,t}))(\sum_{t=1}^Nr(s_{i,t},a_{i,t}))]
  \end{aligned}$$
  
- 高冷的面试官：可以说一下你了解到的基于梯度策略的优化时的小技巧吗？

  答：

  1. **Add a baseline：**为了防止所有的reward都大于0，从而导致每一个stage和action的变换，会使得每一项的概率都会上升。所以通常为了解决这个问题，我们把reward 减掉一项叫做 b，这项 b 叫做 baseline。你减掉这项 b 以后，就可以让 $R(\tau^n)-b$ 这一项， 有正有负。 所以如果得到的 total reward $R(\tau^n)$ 大于 b 的话，就让它的概率上升。如果这个 total reward 小于 b，就算它是正的，正的很小也是不好的，你就要让这一项的概率下降。 如果$R(\tau^n)<b$  ， 你就要让这个 state 采取这个 action 的分数下降 。这样也符合常理。但是使用baseline会让本来reward很大的“行为”的reward变小，降低更新速率。
  2. **Assign suitable credit：** 首先第一层，本来的 weight 是整场游戏的 reward 的总和。那现在改成从某个时间 $t$ 开始，假设这个 action 是在 t 这个时间点所执行的，从 $t$ 这个时间点，一直到游戏结束所有 reward 的总和，才真的代表这个 action 是好的还是不好的；接下来我们再进一步，我们把未来的reward做一个discount，这里我们称由此得到的reward的和为**Discounted Return(折扣回报)** 。
  3. 综合以上两种tip，我们将其统称为**Advantage function**， 用 `A` 来代表 advantage function。Advantage function 是 dependent on s and a，我们就是要计算的是在某一个 state s 采取某一个 action a 的时候，advantage function 有多大。



# Chapter5 Proximal Policy Optimization(PPO) 

## 1 Keywords

- **on-policy(同策略)：** 要learn的agent和环境互动的agent是同一个时，对应的policy。
- **off-policy(异策略)：** 要learn的agent和环境互动的agent不是同一个时，对应的policy。
- **important sampling（重要性采样）：** 使用另外一种数据分布，来逼近所求分布的一种方法，在强化学习中通常和蒙特卡罗方法结合使用，公式如下：$\int f(x) p(x) d x=\int f(x) \frac{p(x)}{q(x)} q(x) d x=E_{x \sim q}[f(x){\frac{p(x)}{q(x)}}]=E_{x \sim p}[f(x)]$  我们在已知 $q$ 的分布后，可以使用上述公式计算出从 $p$ 这个distribution sample x 代入 $f$ 以后所算出来的期望值。
- **Proximal Policy Optimization (PPO)：** 避免在使用important sampling时由于在 $\theta$ 下的 $p_{\theta}\left(a_{t} | s_{t}\right)$ 跟 在  $\theta '$  下的 $p_{\theta'}\left(a_{t} | s_{t}\right)$ 差太多，导致important sampling结果偏差较大而采取的算法。具体来说就是在training的过程中增加一个constrain，这个constrain对应着 $\theta$  跟 $\theta'$  output 的 action 的 KL divergence，来衡量 $\theta$  与 $\theta'$ 的相似程度。

## 2 Questions

- 基于on-policy的policy gradient有什么可改进之处？或者说其效率较低的原因在于？

  答：

  - 经典policy gradient的大部分时间花在sample data处，即当我们的agent与环境做了交互后，我们就要进行policy model的更新。但是对于一个回合我们仅能更新policy model一次，更新完后我们就要花时间去重新collect data，然后才能再次进行如上的更新。

  - 所以我们的可以自然而然地想到，使用off-policy方法使用另一个不同的policy和actor，与环境进行互动并用collect data进行原先的policy的更新。这样等价于使用同一组data，在同一个回合，我们对于整个的policy model更新了多次，这样会更加有效率。

- 使用important sampling时需要注意的问题有哪些。

  答：我们可以在important sampling中将 $p$ 替换为任意的 $q$，但是本质上需要要求两者的分布不能差的太多，即使我们补偿了不同数据分布的权重 $\frac{p(x)}{q(x)}$ 。 $E_{x \sim p}[f(x)]=E_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$ 当我们对于两者的采样次数都比较多时，最终的结果时一样的，没有影响的。但是通常我们不会取理想的数量的sample data，所以如果两者的分布相差较大，最后结果的variance差距（平方级）将会很大。

- 基于off-policy的importance sampling中的 data 是从 $\theta'$ sample 出来的，从 $\theta$ 换成 $\theta'$ 有什么优势？

  答：使用off-policy的importance sampling后，我们不用 $\theta$ 去跟环境做互动，假设有另外一个 policy  $\theta'$，它就是另外一个actor。它的工作是他要去做demonstration，$\theta'$ 的工作是要去示范给 $\theta$ 看。它去跟环境做互动，告诉 $\theta$ 说，它跟环境做互动会发生什么事。然后，借此来训练$\theta$。我们要训练的是 $\theta$ ，$\theta'$  只是负责做 demo，负责跟环境做互动，所以 sample 出来的东西跟 $\theta$ 本身是没有关系的。所以你就可以让 $\theta'$ 做互动 sample 一大堆的data，$\theta$ 可以update 参数很多次。然后一直到 $\theta$  train 到一定的程度，update 很多次以后，$\theta'$ 再重新去做 sample，这就是 on-policy 换成 off-policy 的妙用。

- 在本节中PPO中的KL divergence指的是什么？

  答：本质来说，KL divergence是一个function，其度量的是两个action （对应的参数分别为$\theta$ 和 $\theta'$ ）间的行为上的差距，而不是参数上的差距。这里行为上的差距（behavior distance）可以理解为在相同的state的情况下，输出的action的差距（他们的概率分布上的差距），这里的概率分布即为KL divergence。 


## 3 Something About Interview

- 高冷的面试官：请问什么是重要性采样呀？

  答：使用另外一种数据分布，来逼近所求分布的一种方法，算是一种期望修正的方法，公式是：
  $$\begin{aligned}
  \int f(x) p(x) d x &= \int f(x) \frac{p(x)}{q(x)} q(x) d x \\
  &= E_{x \sim q}[f(x){\frac{p(x)}{q(x)}}] \\
  &= E_{x \sim p}[f(x)]
  \end{aligned}$$
   我们在已知 $q$ 的分布后，可以使用上述公式计算出从 $p$ 分布的期望值。也就可以使用 $q$ 来对于 $p$ 进行采样了，即为重要性采样。

- 高冷的面试官：请问on-policy跟off-policy的区别是什么？

  答：用一句话概括两者的区别，生成样本的policy（value-funciton）和网络参数更新时的policy（value-funciton）是否相同。具体来说，on-policy：生成样本的policy（value function）跟网络更新参数时使用的policy（value function）相同。SARAS算法就是on-policy的，基于当前的policy直接执行一次action，然后用这个样本更新当前的policy，因此生成样本的policy和学习时的policy相同，算法为on-policy算法。该方法会遭遇探索-利用的矛盾，仅利用目前已知的最优选择，可能学不到最优解，收敛到局部最优，而加入探索又降低了学习效率。epsilon-greedy 算法是这种矛盾下的折衷。优点是直接了当，速度快，劣势是不一定找到最优策略。off-policy：生成样本的policy（value function）跟网络更新参数时使用的policy（value function）不同。例如，Q-learning在计算下一状态的预期收益时使用了max操作，直接选择最优动作，而当前policy并不一定能选择到最优动作，因此这里生成样本的policy和学习时的policy不同，即为off-policy算法。

- 高冷的面试官：请简述下PPO算法。其与TRPO算法有何关系呢?

  答：PPO算法的提出：旨在借鉴TRPO算法，使用一阶优化，在采样效率、算法表现，以及实现和调试的复杂度之间取得了新的平衡。这是因为PPO会在每一次迭代中尝试计算新的策略，让损失函数最小化，并且保证每一次新计算出的策略能够和原策略相差不大。具体来说，在避免使用important sampling时由于在 $\theta$ 下的 $p_{\theta}\left(a_{t} | s_{t}\right)$ 跟 在 $\theta'$ 下的 $ p_{\theta'}\left(a_{t} | s_{t}\right) $ 差太多，导致important sampling结果偏差较大而采取的算法。

