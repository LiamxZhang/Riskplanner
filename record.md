# debug

每次debug的时候都报错，在命令行里运行也可能报错，原因好像是conda所建的环境中没有ros2的库;

在vscode debug前，在下面的终端source /opt/ros/humble/setup.bash就可以调试代码了;虽然在.bashrc中已经source过了，但是不知道为什么还要再source一次

然后就可以修改.vscode/launch.json中的python脚本名称和环境地址来调试了

# 在rl/util中新建了一些文件

ppo.py是从sb3库中复制出来的，用于debug

dummy_vec_env.py和subproc_vec_env.py也是从sb3中复制出来的，没做任何修改

env_util.py是从sb3中复制出来的，修改了其中的make_env函数，为了适用于subproc_vec_env的并行环境运行; 需要在make_env中手动import PerceptDroneNav，无法通过函数传参实现

但是subproc_vec_env依然用不了，即使n_env设置为1,程序会在main主程序时新建一个isaacsim仿真窗口，然后在subproc_vec_env新建一个PerceptDroneNav的实例时，再新建一个isaacsim窗口，导致报错

isaac_subproc_vec_env.py是在subproc_vec_env的基础上修改的，将step分成了两步，step1()和step();但是由于subproc_vec_env用不了，所以这个也用不了

isaac_dummy_vec_env.py是仿照dummy_vec_env写的串行运行多个环境的方法;
修改了其step_async()函数，先进行step1()操作，然后QuadrotorIsaacSim().update();之后的step_wait()函数正常获取obs

这种多环境方案也感觉也行，统计了一下平均每次step总用时1300ms,用于step1函数22ms,用于step()函数是0.4ms,剩下的全是用于isaac update的;可以多开几个环境，即使开到50个环境，step部分的时间也才和isaac update的时间差不多

运营PerceptDroneNav_multi_train.py实现，环境写在PerceptionDroneNav里，写了一个集成类

# 多环境的问题
目前存在的问题是,PerceptDroneNavSplitStep环境中，update_trajectory和QuadrotorIsaacSim().update()分开了，原来是一起循环50次;可以明显的看出，分开之后的行为很不稳定，应该是逻辑上有了问题


# 飞控的问题

感觉现在这样飞有点慢


# 奖励部分

我改了奖励和惩罚，现在训练可以开始往前走了，但是目前还是到不了终点。总是out of bounds;因为我修改了这个边界，如果边界太大，飞机会卡在墙角停不下来

而且现在触地和接触桌子都没有惩罚，尤其接触桌子

truncated = self.current_step >= self.max_steps这句话不知道干什么用的，确切的说不知道trucated这个状态对于sb3的ppo来说有什么意义

我本来想修改一下，改为超过max_steps终止terminated;但是max_steps现在设置为1e4有点大，我改小之后，训练几轮会报错;说local map维度有问题

# 程序跑在cpu上
代码似乎运行在CPU下，好像是因为torch的版本有点乱， GPT建议重新装
torch                     2.4.1                    pypi_0    pypi
torchaudio                2.5.1                 py310_cpu    pytorch
torchvision               0.20.1                py310_cpu    pytorch

但是不知道所用cuda到底是12.4还是11.8呢
cuda 11.8
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

cuda 12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

我测试过了，应该是装cuda 12.4版本的，11.8版本对ppo支持有点问题

但是这个修改仅仅对PPO算法的计算有效，PPO维护的计算都用了cuda的tensor, 但是gym环境中从quadrotor中获取的还是cpu tensor, 那这样就没有太大的意义了，可能因为之前是按照cpu写的;

不过这个不是影响训练结果的关键

# 感知模块

我用rviz2可视化了grid map, 似乎只有桌子，也就是地面和墙面都没有感知到，那这个感知不太完全; 飞机触地翻的情况下，没有从obervation中拿到任何地面相关的信息;毕竟也没有输入高度，即使输入了高度，也得从传感器中能看到地面才行吧

# 结论

我觉得可以采用串行的多环境运行，首先这样线程稳定，比单环境运行效果快的多;只是需要修改gym环境中update_trajectory和QuadrotorIsaacSim().update()分开之后的逻辑，需要修改update_trajectory()函数


torch是个小问题，只影响训练速度，并不影响训练结果

另外推荐使用cursor来代替vscode写代码，我跟mingsheng已经用了好几天了，非常好用

https://www.bilibili.com/video/BV1yorUYWEGD?spm_id_from=333.788.videopod.sections&vd_source=b69ac0d2e7f2fe4ba35352ee9d07871b&p=4

可以用composer模式，跟它说帮我修改RL的动作空间，增加输出三维速度；然后他就可以修改，很方面；强于GPT的点是他可以看到所有的代码，能够注意到跨文件的代码上下文