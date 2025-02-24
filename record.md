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
第二是通过debug,检查了每次QuadrotorIsaacSim().update()前后的情况，如果这个循环50次，足够飞机翻好几次了;尤其是如果两个step()之间的action相差比较大的话，会给飞机姿态带来巨大的变化;尝试了循环10次，飞机的行为稳定了许多;

而且感觉planner和飞控配合的不是很好，在step()函数中忽略RL输出的waypoint，把waypoint强行设置在targetpoint,依然飞不过去

测试了一下，感觉是planner生成的轨迹不太好，假如target point是[2, 0, 0]处一个点，planner生成的轨迹是先加速到7.5，然后再目标点处减速到0的那种轨迹，然后飞机要在1m内加速到7.5, 加速度太大，就翻了

不行就设计成端到端的，直接让RL生成推力算了；或者既然输入的状态空间包含速度，那么让RL输出的动作空间也输出一个速度算了

## **另外**,有一个重要的坐标系问题

PARAMS中定义的target_position是和init_position一样定义在世界系下的，但是运行起来之后，从self.quadrotor.state中获取的飞机当前位置是以init_position为原点的;表现形式就是，输入target_position为[1.0, 0.0, 0.0], init_position为[-1, 0.0, 0.0]; 进入程序后一开始获取到的quadrotor.state中的位置是[0.0, 0.0, 0.0], 计算出的target_position的相对坐标是[1,0,0],但是实际上应该是[2,0,0]

这个我改掉了，在init函数中定义
self.target_position = torch.tensor(CONTROL_PARAMS["target_position"] - ROBOT_PARAMS["init_position"], dtype=torch.float32)

但是就算给飞机指示targetposition,planner+飞控依然飞不到那里

# 奖励部分
而且没有碰撞惩罚，只有一个飞机翻过来的惩罚;飞机甚至可以触地平移;感觉不太合理，飞机碰撞桌子之后，只有翻了才会有惩罚，这种间接的reward,飞机不太容易学会避开桌子;而且flip over都没有设置惩罚

计算奖励那里，is_out_of_bounds的penalyt定义的是-50, 结果return了-self.boundary_penalty, 不就是返回的是正数了吗

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

我修改了reward之后效果依然不好，然后检查了代码，影响训练结果的应该有两个主要问题

一个是planner+飞控的结果并不稳定，即使输入ground truth的target point, 也无法稳定到达;可以在step中加一个断点发现，每次step运行的飞机状态都不太稳定

第二是observation中的local grid map不知道对不对， 为什么没有感知到墙面和地面

torch是个小问题，只影响训练速度，并不影响训练结果

另外推荐使用cursor来代替vscode写代码，我跟mingsheng已经用了好几天了，非常好用

https://www.bilibili.com/video/BV1yorUYWEGD?spm_id_from=333.788.videopod.sections&vd_source=b69ac0d2e7f2fe4ba35352ee9d07871b&p=4

可以用composer模式，跟它说帮我修改RL的动作空间，增加输出三维速度；然后他就可以修改，很方面；强于GPT的点是他可以看到所有的代码，能够注意到跨文件的代码上下文