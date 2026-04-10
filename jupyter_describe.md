## DDIM_Inversion Attack

下述四个文件中都采用概念擦除后的模型进行测试.

1.DDIM_Inversion_unerased_Success!!!!!

这里放的是DDIM_Inversion攻击成功后的结果.对于图像到噪声这个反转过程,设置 `guidance_scale=1`,表示图像到噪声这个反演过程不依赖文本.同样噪声到图像的过程也设置 `guidance_scale=1`.最终成功攻击

2.DDIM_Inversion_unerased_Success!!!!! copy

这个文件与上一个文件的唯一不同点是设置 `guidance_scale=3.5.`,表示反转和生成过程都依赖文本,最终的结果就是无法攻击成功

3.DDIM_Inversion_Flux

由于Flux是Rectified Flow模型,不能完全照搬DDIM_Iversion的采样方法.因此这里用了一个Flow Matching Inversion.大体处理与DDIM_Inversion基本一致.攻击成功

4.EDICT

是DDIM的增强版,能达到更高的重构效果,并且要求反转和生成过程的步数严格一致.  代码中有攻击成功的结果展示
