# 【BI思考题】阿里定向广告模型

1. **定向广告和搜索广告的区别**

   搜索广告的实现场景主要是在用户主动去搜索商品的情况下，根据用户搜索的内容推荐给他可能喜欢的商品，如果以$p$来表示用户可能喜欢的商品的概率，则广告的可能表达形式是：
   $$
   p(y=1|ad,user,context)
   $$
   其中ad表示候选广告集，user表示用户特征，context表示上下文场景，设备，事件等等。而整个就表示为上述情况下用户点击广告的条件概率。而搜索广告中，由于用户已经搜索了相关的商品，则候选广告集ad的范围就是和搜索关键词相关。

   但是如果用户在未发出搜索请求时，如何在淘宝主页上呈现用户可能感兴趣的商品内容，进而吸引用户点击，就是定向广告发挥作用的时候了。在这种情况下，候选广告集的范围就是根据用户之前的购买，搜索，点击情况来筛选出的可能感兴趣的商品列表。

2. **定向广告的常见使用模型**

   根据广告的表达形式来看，其实可以抽象为一个二分类问题（点击或不点击，或者说CTR预估）。所以定向广告模型演变过程如下：

   - Logistic Regression：万能的LR；
   - MLR：在LR的基础上又发展出来了Mixed Logistic Regression，其和LR的区别在于MLR可以根据实际情况分别使用不同的LR模型，具有了一定的非线性能力。
   - DNN：由于深度神经网络可以很好的表达非线性关系，并且在CV和NLP场景中的广泛应用，所以在推荐场景中DNN也被拿来处理在大数据场景中的非线性关系。
   - DIN：深度兴趣网络采用了Attention的原理，将用户的兴趣分布根据情况激活并加入到模型训练中。
   - DIEN：DIEN在DIN的基础上优化了兴趣演化层，在Attention中嵌入了序列机制，相对兴趣作用得到了强化。
   - DSIN：这个模型也是在DIN的基础上，将用户行为分为一段一段的session，并用multi-head Attention来获取session内的兴趣。

3. **DIN中Attention的原理机制**

   DIN基本的模型构架还是一个embedding layer和一个MLP组成的，其中embedding layer作用是把稀疏矩阵转移到一个向量空间中，MLP的主要作用是对embedding进行拟合分类输出。在这两个阶段中间，DIN加入了一个activation unit组件，其主要使用的就是attention的机理来计算user feature group的权重。

   ![](https://gitee.com/zhanghang23/picture_bed/raw/master/recommendation_system/taobao_advertisement/din.png)

   Activation Unit的结构如下，则其使用的attention原理可以理解为用户的历史行为对候选ad的权重都是不同的，而用用户历史行为的embedding和候选ad的embedding通过外积的形式来表达相关性。这样在输入之后的MLP时，相关性使得模型可以更好地对候选ad去关注那些有用的历史行为。

   ![](https://gitee.com/zhanghang23/picture_bed/raw/master/recommendation_system/taobao_advertisement/din_au.jpg)

4. **DIEN相对于DIN有哪些创新**

   DIN是中的Attention对于候选ad只是关注了有用的历史行为，但是忽略了一个问题是，用户的历史行为其实是一个时间序列，其会有兴趣的变化，迁移。所以DIEN在DIN的基础上，加入了Interest extractor layer（核心部件是GRU）进行兴趣提取，然后对提取的兴趣加上了一个Interest evolving layer，用改进型AUGRU，并把Attention权重加入到里面让GRU更加关注兴趣的演化，减弱兴趣漂移，之后才输入到MLP中。

   ![](https://gitee.com/zhanghang23/picture_bed/raw/master/recommendation_system/taobao_advertisement/dien.png)

5. **DSIN关于Session的洞察是怎样的，如何对Session兴趣进行表达**

   DSIN提出的问题是，虽然用户具有动态兴趣演化，但是用户使用淘宝的表现为阶段性的，也就是连续搜索一段时间，然后停止，并且每次发生搜索访问时，搜索的物品是很相近，但是两次隔开发生的连续搜索大概率差别很大。DSIN对这种情况，对用户序列建立session，每个session是一个给定时间范围内发生的交互列表，同一个session内的行为高度同构，而跨session之间是异构的。如下图，session在连续浏览裤子，session2就已经在搜索美甲，session3又有变化。但是每个session内部是一个类别的，这就是DSIN发现的问题。

   ![](https://gitee.com/zhanghang23/picture_bed/raw/master/recommendation_system/taobao_advertisement/dsin_session.png)

   所以DSIN对用户的连续行为划分成session，然后用带偏执编码的self attention对每个session进行建模，然后用BI-LSTM捕捉用户不同历史会话兴趣的交互和演变，设计一个局部的活动单元，将他们与目标项聚合起来，形成行为序列的最终表达形式。

   ![](https://gitee.com/zhanghang23/picture_bed/raw/master/recommendation_system/taobao_advertisement/dsin.png)

   首先Session Division Layer 对序列进行分割session。但是session中还是会出现用户的随意行为使的session偏移，所以Session Interest Extractor Layer在每个session中使用multi-head attention来关注session的重点，减轻不相关性行为的影响。Session Interest Extractor Layer对带有session兴趣重点的session使用BI-LSTM来捕捉用户session兴趣的动态演变。之后SIE层和SII层分别表示的session兴趣重点，以及兴趣的演变输入到Session Interest Activating Layer与目标商品之间去计算相关性，给session兴趣设立权重，最终和原始特征合并输入给MLP。

   根据以上描述可以看出，session的兴趣表达主要是通过SIE层的关注兴趣重点，SII层的兴趣演化，并最终通过SIA层分配权重来达到表达兴趣的目的。

6. **如果你来设计淘宝定向广告，会有哪些future work（即下一个阶段的idea）**

   鉴于目前淘宝定向广告已经非常高效了，接下来我感觉可能的下一步走向应该有：

   - 根据用户历史行为和兴趣推断用户可能购买的但从没涉及过的领域。需要建立知识图谱，模型可以认知商品之间的相关关系，并根据时间因素推断可能购买的商品。
   - 有些商品具有强烈的季节性或者时效性（羽绒服，西瓜，凉鞋），定向广告也需要带有时效性（即对兴趣附加一个周期权重），这样定向广告的投放也会有明显的季节性。