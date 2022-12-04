This is the repo the my bachelor's thesis, where I established a differentible model for benchmark Pendulum

it is evaluated with REINFORCE and PPO

Following algorithms are implemented:


1. Original REINFORCE with original OpenAI Gym Pendulum
2. REINFORCE with whitening technique with original OpenAI Gym Pendulum
3. REINFORCE with state-value function as baseline technique with original OpenAI Gym Pendulum
4. Original PPO with original OpenAI Gym Pendulum
5. differentiable model Pendulm with PyTorch
6. REINFORCE with whitening technique with differentiable model Pendulum 
7. PPO with whitening technique with differentiable model Pendulum 
8. Result Postprocessing with Seaborn
9. Generalization of trained models with modification of model parameters

Result:
1. differentiable model provide intuitively additional information in gradient
2. differentiable model can accelerate and stablize the training process of REINFORCE
3. application of differentiable model with PPO is limited because of the inplace operation of backward propagation in PyTorch
4. training could be run with GPU with acceleration (it was not written in my Thesis, it is not tested with experiments yet)  

Further ideas:
1. evaluate in aspect of optimization
2. optimize generalization