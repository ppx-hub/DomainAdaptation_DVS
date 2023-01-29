==修改了之前有问题的地方==

## 1. NCALTECH101的RGB没有对应上dvs的样本

加了这一句

```python
 if args.target_dataset == "NCALTECH101":
            source_input, source_label = source_input_list[tmp_sampler_list], source_label_list[tmp_sampler_list]
```



## 2. 之前的semantic-loss，不仅算了不同域，把自己域也算了

修改方式为，在输出的domain_rbg_list和domain_dvs_list，先计算semantic-loss，再替换domain_rbg_list





## 3. 之前的domain-loss，顺序写反了



```python
# compute domain loss
for b in range(source_input.shape[0]):
    if rd.uniform(0, 1) <= P_Replacement:
        for i in range(len(domain_rbg_list)):
            domain_rbg_list[i][b] = domain_dvs_list[i][b, :, :, :]
```





## 4. 非常奇怪的bug

不加rgb_loss会显存爆炸。。。

```python
loss = loss_rgb + loss_dvs
```





## 5. 关于epochs的设置

设置为300，baseline没问题，但是单阶段的transfer显得有些仓促，效果还不如baseline，考虑600，让transfer发挥作用。
