p2p租车平台业务和airbnb的短租业务是相通的，其第一推荐场景也是搜索推荐：即输入租车时间，取车还车地点，需求车型这三个基本静态特征就可以开始进行搜索推荐了。而第二类推荐场景则是在相应选定搜索结果下的相似车型推荐。鉴于其相似的推荐场景，p2p租车平台的推荐业务也可以借鉴使用airbnb中使用的embedding技术来搭建推荐平台。
## 场景一：相似车型推荐
![p2p_rc](https://gitee.com/zhanghang23/picture_bed/raw/master/recommendation_system/unimportant/%E7%A7%9F%E8%BD%A61.png)
如上图所示，当用户选择了有意进一步了解的车辆页面的下面，会有相似推荐其他类似的可租车辆（也就是airbnb中的”您可能还喜欢“）。
这部分也是使用Embedding的方法，可根据的用户的鼠标点击行为按照单位时间来构建session（session包含每次鼠标点击进入的车辆listing，这些listing会标记为checked（查看）或ordered（下单），当一个session只有ordered listing，则为负样本，当一个session内除了checked listing还有ordered listing，并以ordered listing结尾，则标记为正样本），将session组成数据集，然后用word2vec的skip-gram模型来训练。训练结束后，就可以得到当前页面车辆的listing Embedding。当前的Embedding与其他页面车辆的listing Embedding进行相似度计算，取其Top K作为最后相似车辆推荐的结果。 

## 场景二：搜索推荐
![p2p_rc1](https://gitee.com/zhanghang23/picture_bed/raw/master/recommendation_system/unimportant/%E7%A7%9F%E8%BD%A62.png)
上面的搜索结果来自于p2p租车平台”凹凸租车“，这个场景的搜索步骤也是选择租车城市，然后就是选择品牌和区域了（不过比较奇怪的一点是，在进入到具体车辆页面中才能查看可租时间，这一点设计逻辑有待商榷，毕竟租车的逻辑是先筛选出预期时间内可租的车辆，然后才是查看地点，车型）。
因为p2p租车对于单个的user来说，也存在其ordered session的数据非常少，并且相隔时间长，以及基于地点和出行目的其愿意租赁的车型也会有很大变化（比如公务出差时间短更倾向于商务车，私家旅行时间长趋向于舒适型轿车或旅行车等）。这部分同样也可以参照Airbnb中的user_type和listing_type思想， 按照我初步规划，可根据如下规则进行分桶：  

1.user_type
- 目的城市
- 租车次数
- 历史商务或度假出行比例
- 历史租车每日均价
- 收到的车主打分
- 历史租车期间违规记录数
等等

2.listing_type
- 车辆所在城市
- 车型
- 可乘坐乘客人数
- 每日均价
- 租车用户打分
- 保险规格
- 租车期间车辆出问题次数占总租车次数的比例
等等

然后user_type和listing_type组成一个元组，在同一个向量空间下进行训练得到embedding。也可以用GBDT作为排序模型，原有特征和embedding特征共同输入模型训练。当用户在搜索对应的城市，车辆时，参考特征除了原有特征之外，还会根据当前用户的情况推荐embedding相似度最近的user_type和listing_type，并呈现到推荐结果中。
