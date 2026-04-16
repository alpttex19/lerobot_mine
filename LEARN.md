                                                            
● 三种方法的数学原理详解

---                                                                                                                                                         
一、Diffusion  policy                                                         
1.1 核心思想    
                                                                                                                                                            
我们想建模条件概率分布 $p(\mathbf{a} | \mathbf{o})$，即给定观测 $\mathbf{o}$，动作 $\mathbf{a}$ 的分布。                                                    
                                                                                                                                                            
Diffusion 把这个问题转化为：学习一个去噪过程。

---                                                                                                                                                         
1.2 前向过程（加噪）
                    
定义一个马尔可夫链，逐步给数据加高斯噪声：
                                                                                                                                                            
$$q(\mathbf{x}t | \mathbf{x}{t-1}) = \mathcal{N}(\mathbf{x}t; \sqrt{1-\beta_t}\mathbf{x}{t-1},\ \beta_t \mathbf{I})$$                                       
                                                                                                                                                            
其中 $\beta_t$ 是预定义的噪声schedule（从小到大），$t = 1,...,T$（通常 $T=100$）。                                                                          
                
关键推导：不用真的走 $t$ 步，可以直接从 $\mathbf{x}_0$ 跳到 $\mathbf{x}_t$：                                                                                
                
定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}t = \prod{s=1}^{t} \alpha_s$，则：                                                                              
                
$$\boxed{q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\ \sqrt{\bar{\alpha}_t}\mathbf{x}_0,\ (1-\bar{\alpha}_t)\mathbf{I})}$$                   
                
等价于：                                                                                                                                                    
                
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0,          
\mathbf{I})$$   
                                                                                                                                                            
直觉：$t$ 越大，$\bar{\alpha}_t$ 越小，信号越少，噪声越多。当 $t=T$ 时，$\bar{\alpha}_T \approx 0$，几乎是纯噪声。                                          

---                                                                                                                                                         
1.3 逆向过程（去噪）
                    
真实的逆向分布 $q(\mathbf{x}_{t-1}|\mathbf{x}t)$ 无法直接计算，用神经网络 $\boldsymbol{\epsilon}\theta$ 来近似：
                                                                                                                                                            
$$p_\theta(\mathbf{x}{t-1}|\mathbf{x}t, \mathbf{o}) = \mathcal{N}(\mathbf{x}{t-1};\ \boldsymbol{\mu}\theta(\mathbf{x}_t, t, \mathbf{o}),\ \sigma_t^2        
\mathbf{I})$$                                                                                                                                               
                                                                                                                                                            
其中均值：      

$$\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}t}}\boldsymbol{\epsilon}\theta(\mathbf{x}_t, t,
\mathbf{o})\right)$$
                                                                                                                                                            
---             
1.4 训练目标
            
最大化似然等价于最小化：
                                                                                                                                                            
$$\boxed{\mathcal{L} = \mathbb{E}_{t, \mathbf{x}0, \boldsymbol{\epsilon}}\left[|\boldsymbol{\epsilon} -                                                     
\boldsymbol{\epsilon}\theta(\underbrace{\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}t}\boldsymbol{\epsilon}}{\mathbf{x}_t},\ t,\                
\mathbf{o})|^2\right]}$$                                                                                                                                    
                
即：网络预测的噪声 vs 真实加入的噪声，做 MSE。                                                                                                              

---                                                                                                                                                         
1.5 伪代码      

```python
def train_step(action_0, observation):                                                                                                                      
    # action_0: 真实动作 (batch, action_dim)                                                                                                                
    # observation: 观测特征 (batch, obs_dim)                                                                                                                
                                                                                                                                                            
    # 1. 随机采样时间步                                                                                                                                     
    t = random_int(1, T)  # shape: (batch,)                                                                                                                 
                                                                                                                                                            
    # 2. 采样噪声
    epsilon = randn_like(action_0)  # 标准高斯噪声                                                                                                          
                                                                                                                                                            
    # 3. 计算加噪后的动作（直接跳到第t步）                                                                                                                  
    alpha_bar_t = alpha_bars[t]  # 预计算的 ᾱ_t                                                                                                             
    action_t = sqrt(alpha_bar_t) * action_0 \                                                                                                               
            + sqrt(1 - alpha_bar_t) * epsilon
                                                                                                                                                            
    # 4. 网络预测噪声
    epsilon_pred = network(action_t, t, observation)                                                                                                        
                                                                                                                                                            
    # 5. 计算损失
    loss = mse_loss(epsilon_pred, epsilon)                                                                                                                  
                
    return loss 
    
# ============ 推理（DDPM，100步）============                                                                                                              
def inference(observation):
    # 1. 从纯噪声开始                                                                                                                                       
    action = randn(action_dim)  # x_T
                                                                                                                                                            
    # 2. 逐步去噪                                                                                                                                           
    for t in range(T, 0, -1):  # T=100 down to 1                                                                                                            
        # 预测噪声                                                                                                                                          
        epsilon_pred = network(action, t, observation)
                                                                                                                                                            
        # 计算均值                                                                                                                                          
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]                                                                                                                         
        beta_t = betas[t]
                                                                                                                                                            
        mu = (1 / sqrt(alpha_t)) * (
            action - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_pred                                                                                        
        )                                                                                                                                                   

        # 加随机噪声（t>1时）                                                                                                                               
        if t > 1:
            sigma_t = sqrt(beta_t)                                                                                                                          
            action = mu + sigma_t * randn_like(action)                                                                                                      
        else:
            action = mu                                                                                                                                     
                
    return action  # x_0，最终动作                                                                                                                          
```

---                                                                                                                                                         
---             
二、Flow Matching
                
2.1 核心思想转变
                                                                                                                                                            
Diffusion 的去噪路径是弯曲的随机过程（SDE）。Flow Matching 改为学习一个确定性的速度场，把噪声沿直线"流"到数据。                                             
                                                                                                                                                            
Diffusion:  x_T ~~~随机游走~~→ x_0   (弯曲，需要多步)                                                                                                       
Flow:       x_1 ——直线——→ x_0       (笔直，少步精确)                                                                                                        
                                                                                                                                                            
注意：Flow Matching 习惯用 $t \in [0,1]$，$t=0$ 是数据，$t=1$ 是噪声（和 Diffusion 方向相反）。                                                             
                                                                                                                                                            
---                                                                                                                                                         
2.2 构造插值路径
                                                                                                                                                            
定义从噪声 $\mathbf{x}_1 \sim \mathcal{N}(0,\mathbf{I})$ 到数据 $\mathbf{x}_0$ 的直线插值：
                                                                                                                                                            
$$\boxed{\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0, 1]}$$                                                                            
                                                                                                                                                            
对应的真实速度（这条路径的导数）：                                                                                                                          
                
$$\frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0$$                                                                                                  
                
直觉：每个时刻的速度就是"从数据指向噪声"的向量，是常数！路径是直线。                                                                                        
                
---                                                                                                                                                         
2.3 训练目标    
                                                                                                                                                            
训练神经网络 $\mathbf{v}_\theta$ 去拟合这个速度场：
                                                                                                                                                            
$$\boxed{\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}1}\left[|\mathbf{v}\theta(\mathbf{x}_t, t, \mathbf{o}) - (\mathbf{x}_1 -                      
\mathbf{x}_0)|^2\right]}$$                                                                                                                                  
                                                                                                                                                            
即：网络预测的速度 vs 真实速度（噪声-数据），做 MSE。                                                                                                       

比 Diffusion 更简洁——目标是固定的向量，不是依赖 $t$ 的复杂函数。                                                                                            
                
---                                                                                                                                                         
2.4 推理（ODE求解）
                                                                                                                                                            
推理时，从噪声出发，用网络预测的速度场做数值积分（ODE求解）：
                                                                                                                                                            
$$\mathbf{x}_{t - \Delta t} = \mathbf{x}t - \mathbf{v}\theta(\mathbf{x}_t, t, \mathbf{o}) \cdot \Delta t$$                                                  
                                                                                                                                                            
因为路径接近直线，步长可以很大，步数可以很少（5-10步就够）。                                                                                                
                
---                                                                                                                                                         
2.5 伪代码      
```python
# ============ 训练 ============
def train_step(action_0, observation):                                                                                                                      
    # 1. 采样噪声（x_1）
    action_1 = randn_like(action_0)                                                                                                                         
                
    # 2. 采样时间步                                                                                                                                         
    t = random_uniform(0, 1)  # 连续值
                                                                                                                                                            
    # 3. 直线插值，构造 x_t                                                                                                                                 
    action_t = (1 - t) * action_0 + t * action_1                                                                                                            
                                                                                                                                                            
    # 4. 真实速度（常数！）                                                                                                                                 
    velocity_target = action_1 - action_0
                                                                                                                                                            
    # 5. 网络预测速度                                                                                                                                       
    velocity_pred = network(action_t, t, observation)
                                                                                                                                                            
    # 6. 损失   
    loss = mse_loss(velocity_pred, velocity_target)

    return loss                                                                                                                                             

# ============ 推理（N步ODE，通常N=5~10）============                                                                                                       
def inference(observation, N=10):
    # 1. 从纯噪声开始（x_1）                                                                                                                                
    action = randn(action_dim)                                                                                                                              

    # 2. 时间从1积分到0（噪声→数据）                                                                                                                        
    dt = 1.0 / N
    for t in linspace(1.0, dt, N):  # 1.0, 0.9, 0.8, ..., 0.1                                                                                               
        # 预测速度                                                                                                                                          
        velocity = network(action, t, observation)                                                                                                          
                                                                                                                                                            
        # 欧拉步进（向 t=0 方向走）                                                                                                                         
        action = action - velocity * dt
                                                                                                                                                            
    return action  # x_0，最终动作
    
```

---                                                                                                                                                         
2.6 为什么比 Diffusion 快

Diffusion 路径（弯曲）:
x₁₀₀ → x₉₉ → x₉₈ → ... → x₀                                                                                                                                 
数值积分误差大，必须小步走                                                                                                                                  
                                                                                                                                                            
Flow Matching 路径（直线）:                                                                                                                                 
x₁.₀ ————————————→ x₀                                                                                                                                       
路径几乎是直线，大步走误差小                                                                                                                                

数学上：路径曲率越小，ODE 求解越精确，需要的步数越少。                                                                                                      
                
---                                                                                                                                                         
---             
三、Consistency Model

3.1 核心思想

基于 Diffusion 的去噪轨迹，定义一个函数 $f$：                                                                                                               

$$\boxed{f(\mathbf{x}_t, t, \mathbf{o}) = \mathbf{x}_0 \quad \forall t \in [0, T]}$$                                                                        
                
即：不管输入哪个时间步的带噪动作，直接输出干净动作 $\mathbf{x}_0$。                                                                                         
                
这个函数必须满足一致性条件：同一条轨迹上所有点映射到同一个 $\mathbf{x}_0$。                                                                                 
                
---                                                                                                                                                         
3.2 网络参数化  
            
直接用网络预测 $\mathbf{x}_0$ 会导致 $t \to 0$ 时退化（此时 $\mathbf{x}_t \approx \mathbf{x}_0$，预测太容易没意义）。
                                                                                                                                                            
实际用加权组合参数化：                                                                                                                                      
                                                                                                                                                            
$$f_\theta(\mathbf{x}t, t, \mathbf{o}) = c\text{skip}(t) \cdot \mathbf{x}t + c\text{out}(t) \cdot F_\theta(\mathbf{x}_t, t, \mathbf{o})$$                   
                
其中：                                                                                                                                                      
- $c_\text{skip}(t)$：$t \to 0$ 时趋向1（直接复制输入），$t \to T$ 时趋向0
- $c_\text{out}(t)$：$t \to 0$ 时趋向0，$t \to T$ 时趋向1                                                                                                   
- $F_\theta$：实际的神经网络                             
                                                                                                                                                            
边界条件自动满足：$f_\theta(\mathbf{x}_0, 0, \mathbf{o}) = \mathbf{x}_0$
                                                                                                                                                            
---                                                                                                                                                         
3.3 训练目标（Consistency Training）                                                                                                                        
                                                                                                                                                            
取相邻两个时间步 $t$ 和 $t+\Delta t$，要求它们的预测一致：
                                                                                                                                                            
$$\boxed{\mathcal{L} = \mathbb{E}{t, \mathbf{x}0, \boldsymbol{\epsilon}}\left[\lambda(t) \cdot d\left(f\theta(\mathbf{x}{t+\Delta t}, t+\Delta t,           
\mathbf{o}),\ f_{\theta^-}(\mathbf{x}_t, t, \mathbf{o})\right)\right]}$$                                                                                    
                                                                                                                                                            
其中：          
- $d(\cdot, \cdot)$：距离函数（MSE 或 LPIPS）
- $\lambda(t)$：权重函数                                                                                                                                    
- $\theta^-$：EMA（指数移动平均）参数，作为稳定的"目标网络"，不直接更新
                                                                                                                                                            
注意：$\mathbf{x}_{t+\Delta t}$ 和 $\mathbf{x}_t$ 来自同一条轨迹（同一个 $\mathbf{x}_0$ 加噪得到），所以理想情况下 $f$ 对两者的输出应该相同。               
                                                                                                                                                            
---                                                                                                                                                         
3.4 多步采样（提升质量）                                                                                                                                    
                                                                                                                                                            
一步采样（最快）:
x_T → f(x_T) = x_0_rough  完成                                                                                                                              
                                                                                                                                                            
多步采样（质量更好）:                                                                                                                                       
x_T → f(x_T) = x_0'                                                                                                                                         
    → 对 x_0' 重新加噪到 x_{T/2}    # 加中等噪声                                                                                                            
    → f(x_{T/2}) = x_0''            # 再次预测，更精细                                                                                                      
    → 对 x_0'' 重新加噪到 x_{T/4}                                                                                                                           
    → f(x_{T/4}) = x_0'''           # 继续精化                                                                                                              
                                                                                                                                                            
每一轮"加噪再去噪"都在精化细节，但总步数远少于 Diffusion。                                                                                                  
                                                                                                                                                            
---                                                                                                                                                         
3.5 伪代码      

```python
# ============ 训练（Consistency Training）============
def train_step(action_0, observation):
    # 1. 采样相邻时间步对                                                                                                                                   
    t = random_int(1, T-1)
    t_next = t + 1                                                                                                                                          
                
    # 2. 采样同一噪声                                                                                                                                       
    epsilon = randn_like(action_0)
                                                                                                                                                            
    # 3. 构造同一轨迹上的两点                                                                                                                               
    action_t      = sqrt(alpha_bars[t])      * action_0 \
                + sqrt(1 - alpha_bars[t])      * epsilon                                                                                                  
    action_t_next = sqrt(alpha_bars[t_next]) * action_0 \
                + sqrt(1 - alpha_bars[t_next]) * epsilon                                                                                                  
                
    # 4. 两个时间步的预测                                                                                                                                   
    # 用 EMA 参数预测 t（稳定目标）
    pred_t      = f_ema(action_t,      t,      observation)                                                                                                 
    # 用当前参数预测 t+1（被优化）
    pred_t_next = f_theta(action_t_next, t_next, observation)                                                                                               
                                                                                                                                                            
    # 5. 一致性损失（两个预测应该相同）                                                                                                                     
    loss = mse_loss(pred_t_next, pred_t.detach())                                                                                                           
                                                                                                                                                            
    # 6. 更新 EMA 参数                                                                                                                                      
    update_ema(f_ema, f_theta, decay=0.999)
                                                                                                                                                            
    return loss 

# ============ 推理 ============                                                                                                                            
def inference_one_step(observation):
    # 直接一步                                                                                                                                              
    action_T = randn(action_dim)
    action_0 = f_theta(action_T, T, observation)                                                                                                            
    return action_0
                                                                                                                                                            
def inference_multi_step(observation, steps=[T, T//2, T//4]):                                                                                               
    action = randn(action_dim)                                                                                                                              
                                                                                                                                                            
    for t in steps:
        # 预测 x_0
        action_0_est = f_theta(action, t, observation)                                                                                                      

        if t != steps[-1]:                                                                                                                                  
            # 重新加噪到下一个时间步（不是纯噪声，是中等噪声）
            t_next = steps[steps.index(t) + 1]                                                                                                              
            epsilon = randn_like(action_0_est)                                                                                                              
            action = sqrt(alpha_bars[t_next]) * action_0_est \                                                                                              
                    + sqrt(1 - alpha_bars[t_next]) * epsilon                                                                                                 
                
    return action_0_est
```

---
四、三者对比总结
                                                                                                                                                            
训练目标对比
                                                                                                                                                            
Diffusion:      min || ε - ε_θ(x_t, t, o) ||²
                    预测噪声                                                                                                                               
                                                                                                                                                            
Flow Matching:  min || (x₁-x₀) - v_θ(x_t, t, o) ||²                                                                                                         
                    预测速度                                                                                                                               
                                                                                                                                                            
Consistency:    min d( f_θ(x_{t+1}, t+1, o), f_θ-(x_t, t, o) )                                                                                              
                    相邻时间步预测一致
                                                                                                                                                            
推理路径对比    

Diffusion（100步，随机）:
噪声 ~~→ ~~→ ~~→ ~~→ ~~→ ~~→ 动作                                                                                                                          
                                                                                                                                                            
Flow Matching（5-10步，确定性）:                                                                                                                            
噪声 ——→ ——→ ——→ 动作                                                                                                                                      
                                                                                                                                                            
Consistency（1-3步，跳跃）:                                                                                                                                 
噪声 ════════════→ 动作
                                                                                                                                                            
核心公式一览

| 方法 | 插值/路径 | 训练目标 | 推理步数 |
|------|---------|---------|---------|
| Diffusion | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ | 预测噪声 $\boldsymbol{\epsilon}$ | 100步 |
| Flow Matching | $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ | 预测速度 $\mathbf{x}_1 - \mathbf{x}_0$ | 5-10步 |
| Consistency | 同 Diffusion | 相邻步预测一致 | 1-3步 | 

一句话理解      
        
- Diffusion：学会每一小步怎么去噪，走完100步到终点
- Flow Matching：学会速度方向，沿直线快速走到终点                       
- Consistency：学会从任意位置直接看到终点在哪里  