# Attack  Concept Erased Diffusion Model via DDIM_inversion
## 前言

​	看到这篇文章TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models能被cvpr'26接受，我们(AISEC in XJTU)的内心其实是欣喜的。因为我们在2025年11月左右观察到了类似的现象。虽然处理方法与问题的思考角度略有不同，但出发点是完全相同的。

​	具体而言，我们也尝试了使用DDIM_Inversion这项技术攻击Concept Erased Diffusion Model，并且我们也尝试将condition条件设为空(cfg=1)。

## 思路

​	想要捋顺我们的思路，这件事要从这篇文章谈起When Are Concepts Erased From Diffusion Models?（下文简称`Where`），这篇文章的三作正是Erasing Concepts from Diffusion Models的一作。并且这些人都来自DavidBau Lab.

​	`Where`这篇文章对概念擦除后的扩散模型(下文简称`ESD`)，提出了三种探针(其实是四种，但是我们暂不讨论第四种，因为我认为它有点牵强，还不足以称为probe)，从而证明当前的`ESD`并没有完完全全将概念抹除。三种探针分别为:

​	1. 文本探针

​	将文本经过一系列特殊处理，然后丢给Diffusion Model，从而测试是否能够唤醒被擦除的概念(比如裸体，由于绝大部分的ESD模型都要在裸图上进行评测，因此下文中被擦除的概念我们都以“裸”为例)。这种文本攻击方法其实早就被探索过，比如MMA(语义余弦相似度)，textural Inversion(利用语义空间的冗余性)。虽然MMA和textual Inversion最初并没有用于攻击`ESD`，但后续将其作为验证ESD鲁棒性的工作也有不少

​	2.图像上下文探针

​	这个探针可以分为两块

​	2.1图像修复

​	将一个完整图像扣出去一块(比如裸体部分),丢给diffusion inpainting pipeline，如果能完成裸体部分的修复，则证明ESD不鲁棒

​	2.2提取扩散的前几步

​	对于一个没有进行概念擦除的扩散模型(下文简称`non-ESD`),让它生成一个裸体图像，但是仅提取扩散的前几步(比如完整为50步，仅提取第3步的噪声，设为`NSFW-Noise-3`),将`NSFW-Noise-3`喂给ESD，如果能生成裸体图像，则证明其不鲁棒

​	3.采样轨迹

​	文章猜测ESD并没有实现鲁棒性擦除，某一个概念只是在某种引导下发生了漂移，因此如果向采样轨迹中注入一些随机性，没准就能找到发生了漂移的概念所在的Distribution。结果也较为成功



​	从时间线上来看，该文章的方法2和方法3，才是**首个**，Text-Free 的探针(攻击手段)



​	其中，我们对2.2比较感兴趣，我们知道，根据cfg调度相关的理论，Diffusion的前几步将很大程度上决定了图像的Structure，所以此时的噪声是一种`Conditional`的，或者说带有信息的非纯噪声。   此时就非常容易想到另一种噪声构造方式：DDIM-Inversion

​	接下来我们就要开始构思实验了

​	1. exp1：给定一个已知的裸图，设为`n-image`，将他送进`non-ESD`，进行DDIM-Inversion的forward操作，得到一个Conditional的噪声(设为`ddim_noise`)，将`ddim_noise`喂给一个ESD模型，进行DDIM-reverse采样看看是否能生成裸图。

​	在做实验前，我们已经猜到了结果，肯定是不行的，因为DDIM要求forward和reverse必须采用同一个UNet，因为DDIM这个方法本质上在干的事情是，摒弃DDPM的马尔科夫性，用一个ODE来完成采样，从而提高采样效率。采用不同的UNet意味着原始的Distribution已经发生了偏移，一定不会有好的效果。 结果果真如此
<img width="1239" height="480" alt="image" src="https://github.com/user-attachments/assets/2b7b93c6-3f5d-4142-9cab-c4354854ca16" />
2.1 exp2:由于(1)，我们接下来尝试了，直接让`ESD`完成DDIM_forward 和DDIM_reverse. 

​	做实验前，我们依旧猜测了结果：一定能够恢复出原图像，但在视觉上可能有些局部失真(比如手部模糊或者多指），失真大概率来自VAE，降低失真的最笨方法就是直接增大分辨率(如512×512---->1024×1024)，或者采用其他采样方法

​	结果果真如此，我们的猜想完全正确。但是对于概念擦除来讲，我们认为这是**没有意义**的，最后解释。

<img width="875" height="458" alt="image" src="https://github.com/user-attachments/assets/99092b6b-c4d7-471f-8edd-2b0bb7e59385" />

2.2 exp3采用其他的采样方法

​	“增大分辨率”这种方法太没有技术性了。与TINA直接重新训练DDIM trajectory的方式不同，我们直接采用其他的采样方式`EDICT`,因为我们认为，重训一个DDIM trajectory有点画蛇添足的意味。

​	因为DDIM本质上在完成“用ODE拟合DDPM的类SDE轨迹(误差为$o(t^2)$”这件事情，从而实现100step--->few step的提升。而现在的主流方法flow mathcing ,rectified flow(Flux，Sana等产品)，采用速度场的训练方式(v-prediction)，在采样阶段直接就是ODE。        从diffusion领域的发展时间线上来讲，在2025年，为DDIM训一个增强采样的方式完全没必要。

​	对于$\epsilon$-prediction的方法，我们采用了`EDICT`,对于v-prediction的方法，我们采用了`flow-reverse`

​	3. exp4 考虑$cfg \neq 1$的情况

​	DDIM_Inversion这个设计之初时为了完成图像编辑的任务，比如，将“草坪上有一只小狗”这张图像编辑为"草坪上有一只小猫"。所以我们尝试利用它完成对`n-image`的编辑。

​	具体实现上，将`n-image`喂给`ESD`，设cfg=1,完成DDIM-forward，得到噪声`ddim_noise`.

​	将`ddim_noise`喂给ESD，设$cfg \neq 1$,然后给定`prompt=a naked girl is dancing`或者其他符合语境的prompt，看看ESD是否能够完成生成。

​	在这次实验中，我们没有很明确的预期结果。实验效果如下

​	失真的程度非常高，但是这种失真是可以通过cfg的线性调度来弥补(比如前几步采样用cfg=1，从而保证整体上与原始图像保持一致，后几步采用cfg=10,保证有编辑效果。或者采用linear递增，sublinear递增的调度)
<img width="1170" height="597" alt="image" src="https://github.com/user-attachments/assets/db859420-d8cf-4c40-8fd3-a126b74ab6b3" />
	4.exp5 flow reverse

​	上述实验的攻击对象都是比较low的sdv1-4，我们尝试攻击DiT-based Diffuison Model，攻击对象是EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers，这篇文章用了LoRA微调Flux-dev。

​	我们实现了flow reverse的算法，最终效果如下
<img width="910" height="524" alt="image" src="https://github.com/user-attachments/assets/67fdfc42-096f-49d7-9c85-649428c9f632" />
	同样我们还设置了线性调度cfg的方式，完成了`prompt=a naked girl is dancing/laughing`等符合语境的Editing。
## 小结
(1)我们与TINA的出发点一样，将DDIM_Inversion作为一个鲁棒性探针，并且将condition设置为空(tina中c=null,我们是cfg=1，二者完全等价)

​	(2)TINA选择了训一个模型，修正DDIM_inversion的中间变量$z_t$,

​	我们站在“解ODE”的角度思考问题，对于$\epsilon$-prediction的方法采用`EDCIT`(理论误差为o(t^3)-o(t^3))重构原图像，对于v-prediction的模型采用`flow-reverse`重构原图像

​	(3)我们还在ESD上实现了NSFW(Erased Concepts)图像的editing任务
## 对于“没有意义”的解释   ※※※※※※※
如果您捋顺了思路，上述这个五个实验的设计其实在十分钟内就能完成。我们也是这样，由于全程不需要train模型，整个5个exp仅仅是我们通宵一晚上的结果。(当然仅限于视觉效果，我们没有跑任何定量指标)

​	但在事后，我们对这次的实验进行了很大程度上的**自我否定**。

​	首先明确一个点，Diffusion Model不是一个end2end模型，对于一个pretrained Diffuison Model来讲，完成各种方式的采样(epsilon-prediction有DDPM_sampling,DDIM_sampling等，v-prediction有eular-maruyama Sampling，Huen‘s Sampling )是非常正常的事情。那么如果给定了`ddim_noise`, pretrained diffusion model一定能够恢复原图，这件事情其实是与是否进行了概念擦除无关。

​	更具体一点，我们上述的所有实验其实是在问`“这个Diffusion model有没有Encode+Decode能力？”`而不是在问`“这个ESD到底是否鲁棒?"`。

​	这也是后续没有将该实验继续向定量指标推进的根本原因。
	从这个角度出发，我们完成的Editing任务其实也在混淆上述这两个问题。如果要进行text-free的攻击，我们应该做的事情应该类似于：假如ESD擦除的对象是**动物**，我们给定一张"狗"的图像，然后判断它能不能生成"猫","狮子"，"大象"等其他**动物**.
