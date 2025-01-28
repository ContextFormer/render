

We adjust the weighting of the goal velocity and employ SAC (Soft Actor-Critic) to train the policy, subsequently collecting diverse datasets. We then utilize contrastive learning to train the information-static value**z_1,z_2,...z_6** and subsequently train our ContextFormer model. Contextformer can distinguish strategies with significantly different speeds, but it cannot differentiate strategies that are relatively similar.

![image](https://github.com/ContextFormer/render/blob/main/che_multi_velocity.png)

We reformualte the reward function from r=x to r=(x^2+y^2)^0.5, and subsequently train SAC to gather offline datasets. Following this, we employ contrastive learning to train the information-static value．　Meanwhile, we proceed to train our ContextFormer model. 

![image](https://github.com/ContextFormer/render/blob/main/ant_left.png)

![image](https://github.com/ContextFormer/render/blob/main/ant_left_sourth.png)
