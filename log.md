1. 用原配置训练多角度正弦域
   configs/sino_multiview_v12to24.yaml
   训练采样：random_group

2. 用新配置生成正弦域结果
   configs/sino_multiview_v12to24_eval_interval.yaml
   生成采样：fixed_interval

3. 对 fixed_interval 的正弦域结果跑 FISTA

4. 图像域使用 fixed_interval 生成出来的 FISTA 数据训练/测试