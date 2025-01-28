

We adjust the weighting of the goal velocity and employ SAC (Soft Actor-Critic) to train the policy, subsequently collecting diverse datasets. We then utilize contrastive learning to train the information-static value and subsequently train our ContextFormer model. Contextformer can distinguish strategies with significantly different speeds, but it cannot differentiate strategies that are relatively similar.

![image](https://github.com/ContextFormer/render/blob/main/che_multi_velocity.png)

We reformualte the reward function from r=x to r=(x^2+y^2)^0.5, then training SAC to collect offline datasets. And then train the information static value via contrasive learning, and then train our ContextFormer. 

![image](https://github.com/ContextFormer/render/blob/main/ant_left.png)
![image](https://github.com/ContextFormer/render/blob/main/ant_left_sourth.png)
