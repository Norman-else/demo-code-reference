# AWS Personalize 新品推荐系统设计文档

## 1. 概述

本文档阐述如何使用AWS Personalize实现一个基于商品被不同商店购买比率的新品推荐系统。该系统旨在为商店推荐尚未购买的新商品，基于其他商店对这些商品的购买模式。

### 1.1 业务需求
- 为商店推荐新商品（商店尚未购买的商品）
- 基于商品被其他商店购买的比率进行推荐
- 过滤掉商店已经购买过的商品
- 数据仅包含：商店ID、商品ID、历史采购信息

### 1.2 技术目标
- 利用AWS Personalize的协同过滤能力
- 实现实时推荐API
- 确保推荐结果的相关性和新颖性

## 2. 数据模型设计

### 2.1 数据架构

由于只有商店ID、商品ID和历史采购信息，我们采用简化的数据模型：

#### 交互数据集 (Interactions Dataset)
```json
{
  "USER_ID": "store_id",
  "ITEM_ID": "product_id", 
  "TIMESTAMP": "purchase_timestamp",
  "EVENT_TYPE": "purchase",
  "EVENT_VALUE": 1
}
```

#### 数据字段说明
- **USER_ID**: 商店ID（在Personalize中作为用户）
- **ITEM_ID**: 商品ID
- **TIMESTAMP**: 采购时间戳（Unix时间戳格式）
- **EVENT_TYPE**: 固定值"purchase"
- **EVENT_VALUE**: 固定值1（表示采购事件）

### 2.2 数据预处理要求

1. **时间戳标准化**
   ```python
   # 转换为Unix时间戳
   timestamp = int(datetime.strptime(date_string, '%Y-%m-%d').timestamp())
   ```

2. **数据清洗**
   - 去除重复的采购记录
   - 确保商店ID和商品ID的唯一性
   - 验证时间戳的有效性

3. **数据格式**
   ```csv
   USER_ID,ITEM_ID,TIMESTAMP,EVENT_TYPE,EVENT_VALUE
   store_001,product_001,1609459200,purchase,1
   store_001,product_002,1609545600,purchase,1
   store_002,product_001,1609632000,purchase,1
   ```

## 3. AWS Personalize 配置

### 3.1 数据集组 (Dataset Group) 设置

```python
import boto3

personalize = boto3.client('personalize')

# 创建数据集组
dataset_group_response = personalize.create_dataset_group(
    name='store-product-recommendation',
    description='Store product recommendation based on purchase patterns'
)
```

### 3.2 数据集 Schema 定义

```json
{
    "type": "record",
    "name": "Interactions",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {
            "name": "USER_ID",
            "type": "string"
        },
        {
            "name": "ITEM_ID", 
            "type": "string"
        },
        {
            "name": "TIMESTAMP",
            "type": "long"
        },
        {
            "name": "EVENT_TYPE",
            "type": "string"
        },
        {
            "name": "EVENT_VALUE",
            "type": "float"
        }
    ],
    "version": "1.0"
}
```

### 3.3 推荐算法选择

基于业务需求，推荐使用以下算法：

#### 主要推荐算法: User-Personalization
```python
recipe_arn = 'arn:aws:personalize:::recipe/aws-user-personalization'
```

**选择理由:**
- 支持实时学习
- 能够处理隐式反馈（采购行为）
- 支持冷启动问题
- 自动考虑商品流行度

#### 备选算法: SIMS (Similar Items)
```python
recipe_arn = 'arn:aws:personalize:::recipe/aws-sims'
```

**使用场景:**
- 当需要基于商品相似性推荐时
- 作为User-Personalization的补充

### 3.4 解决方案版本配置

```python
# 创建解决方案
solution_response = personalize.create_solution(
    name='store-new-product-recommendation',
    datasetGroupArn=dataset_group_arn,
    recipeArn='arn:aws:personalize:::recipe/aws-user-personalization',
    solutionConfig={
        'algorithmHyperParameters': {
            'bptt': '32',
            'hidden_dimension': '149', 
            'recency_mask': 'true'
        },
        'eventValueThreshold': '0.5',
        'featureTransformationParameters': {
            'max_hist_len_percentile': '0.99',
            'min_hist_len_percentile': '0.0'
        }
    }
)

# 训练解决方案版本
solution_version_response = personalize.create_solution_version(
    solutionArn=solution_arn
)
```

## 4. 推荐过滤实现

### 4.1 过滤器创建

为确保不推荐商店已购买的商品，需要创建过滤器：

```python
# 创建过滤器表达式
filter_expression = '''
EXCLUDE ItemID WHERE Interactions.EVENT_TYPE IN ("purchase")
'''

# 创建过滤器
filter_response = personalize.create_filter(
    name='exclude-purchased-items',
    datasetGroupArn=dataset_group_arn,
    filterExpression=filter_expression
)
```

### 4.2 实时推荐配置

```python
# 创建营销活动
campaign_response = personalize.create_campaign(
    name='store-new-product-campaign',
    solutionVersionArn=solution_version_arn,
    minProvisionedTPS=1,
    campaignConfig={
        'itemExplorationConfig': {
            'explorationWeight': '0.3',
            'explorationItemAgeCutOff': '30'
        }
    }
)
```

## 5. API 实现

### 5.1 推荐服务 API

```python
import boto3
import json
from typing import List, Dict

class StoreProductRecommendationService:
    def __init__(self):
        self.personalize_runtime = boto3.client('personalize-runtime')
        self.campaign_arn = 'arn:aws:personalize:region:account:campaign/store-new-product-campaign'
        self.filter_arn = 'arn:aws:personalize:region:account:filter/exclude-purchased-items'
    
    def get_recommendations(self, store_id: str, num_results: int = 10) -> List[Dict]:
        """
        为指定商店获取新商品推荐
        
        Args:
            store_id: 商店ID
            num_results: 推荐商品数量
            
        Returns:
            推荐商品列表
        """
        try:
            response = self.personalize_runtime.get_recommendations(
                campaignArn=self.campaign_arn,
                userId=store_id,
                numResults=num_results,
                filterArn=self.filter_arn
            )
            
            recommendations = []
            for item in response['itemList']:
                recommendations.append({
                    'product_id': item['itemId'],
                    'score': item['score'],
                    'reason': 'Based on similar stores purchasing patterns'
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []
    
    def get_similar_items(self, product_id: str, num_results: int = 10) -> List[Dict]:
        """
        获取相似商品推荐
        
        Args:
            product_id: 商品ID
            num_results: 推荐数量
            
        Returns:
            相似商品列表
        """
        try:
            response = self.personalize_runtime.get_recommendations(
                campaignArn=self.sims_campaign_arn,  # SIMS算法的campaign
                itemId=product_id,
                numResults=num_results
            )
            
            similar_items = []
            for item in response['itemList']:
                similar_items.append({
                    'product_id': item['itemId'],
                    'score': item['score']
                })
            
            return similar_items
            
        except Exception as e:
            print(f"Error getting similar items: {str(e)}")
            return []
```

### 5.2 批量推荐实现

```python
def get_batch_recommendations(self, store_ids: List[str]) -> Dict[str, List[Dict]]:
    """
    批量获取多个商店的推荐
    
    Args:
        store_ids: 商店ID列表
        
    Returns:
        商店推荐字典
    """
    batch_results = {}
    
    # 创建批量推荐作业
    job_input = {
        'inputPath': 's3://your-bucket/batch-input/',
        'outputPath': 's3://your-bucket/batch-output/'
    }
    
    # 准备输入数据
    input_data = []
    for store_id in store_ids:
        input_data.append({
            'userId': store_id
        })
    
    # 执行批量推荐
    batch_job_response = self.personalize.create_batch_inference_job(
        jobName=f'batch-recommendations-{int(time.time())}',
        solutionVersionArn=self.solution_version_arn,
        filterArn=self.filter_arn,
        jobInput=job_input,
        jobOutput=job_input,
        roleArn='arn:aws:iam::account:role/PersonalizeBatchRole',
        batchInferenceJobConfig={
            'itemExplorationConfig': {
                'explorationWeight': '0.3'
            }
        }
    )
    
    return batch_job_response
```

## 6. 部署架构

### 6.1 系统架构图

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│   S3 Bucket      │───▶│  AWS Personalize│
│  (Purchase Data)│    │  (Training Data) │    │   (ML Models)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │◀───│   API Gateway    │◀───│   Lambda        │
│   (Store UI)    │    │   (REST API)     │    │  (Recommendation│
└─────────────────┘    └──────────────────┘    │   Service)      │
                                                └─────────────────┘
```

### 6.2 Lambda 函数实现

```python
import json
import boto3
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda函数处理推荐请求
    """
    try:
        # 解析请求参数
        store_id = event['pathParameters']['storeId']
        num_results = int(event.get('queryStringParameters', {}).get('limit', 10))
        
        # 初始化推荐服务
        recommendation_service = StoreProductRecommendationService()
        
        # 获取推荐
        recommendations = recommendation_service.get_recommendations(
            store_id=store_id,
            num_results=num_results
        )
        
        # 返回结果
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'store_id': store_id,
                'recommendations': recommendations,
                'total_count': len(recommendations)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
```

## 7. 数据更新策略

### 7.1 实时数据更新

```python
def update_interaction_data(store_id: str, product_id: str, timestamp: int):
    """
    实时更新采购数据
    """
    personalize_events = boto3.client('personalize-events')
    
    try:
        response = personalize_events.put_events(
            trackingId='your-tracking-id',
            userId=store_id,
            sessionId=f'session-{store_id}-{int(time.time())}',
            eventList=[
                {
                    'eventId': f'event-{store_id}-{product_id}-{timestamp}',
                    'eventType': 'purchase',
                    'eventValue': 1.0,
                    'itemId': product_id,
                    'sentAt': timestamp
                }
            ]
        )
        return response
        
    except Exception as e:
        print(f"Error updating interaction data: {str(e)}")
        return None
```

### 7.2 批量数据更新

```python
def batch_update_from_s3():
    """
    从S3批量更新训练数据
    """
    # 1. 准备新的训练数据
    # 2. 上传到S3
    # 3. 创建新的数据集导入作业
    
    import_job_response = personalize.create_dataset_import_job(
        jobName=f'import-job-{int(time.time())}',
        datasetArn=dataset_arn,
        dataSource={
            'dataLocation': 's3://your-bucket/updated-interactions.csv'
        },
        roleArn='arn:aws:iam::account:role/PersonalizeRole'
    )
    
    return import_job_response
```

## 8. 监控与优化

### 8.1 性能监控指标

1. **推荐准确性指标**
   - 点击率 (CTR)
   - 转化率
   - 覆盖率

2. **系统性能指标**
   - API响应时间
   - 吞吐量
   - 错误率

### 8.2 A/B测试框架

```python
def ab_test_recommendations(store_id: str, test_variant: str = 'A'):
    """
    A/B测试不同推荐策略
    """
    if test_variant == 'A':
        # 使用User-Personalization算法
        campaign_arn = 'arn:aws:personalize:region:account:campaign/variant-a'
    else:
        # 使用SIMS算法
        campaign_arn = 'arn:aws:personalize:region:account:campaign/variant-b'
    
    # 获取推荐并记录测试数据
    recommendations = get_recommendations_with_campaign(store_id, campaign_arn)
    
    # 记录测试指标
    log_ab_test_metrics(store_id, test_variant, recommendations)
    
    return recommendations
```

### 8.3 模型重训练策略

```python
def retrain_model_schedule():
    """
    定期重训练模型
    """
    # 检查数据增长情况
    data_growth = check_data_growth_rate()
    
    if data_growth > 0.1:  # 数据增长超过10%
        # 触发重训练
        new_solution_version = personalize.create_solution_version(
            solutionArn=solution_arn
        )
        
        # 等待训练完成后更新campaign
        update_campaign_with_new_version(new_solution_version['solutionVersionArn'])
```

## 9. 成本优化

### 9.1 成本分析

1. **训练成本**
   - 按训练小时计费
   - 建议每周重训练一次

2. **推理成本**
   - 按TPS计费
   - 根据业务需求调整最小TPS

3. **存储成本**
   - S3存储训练数据
   - 建议使用生命周期策略

### 9.2 成本优化建议

```python
# 动态调整TPS
def adjust_campaign_capacity(current_usage: float):
    """
    根据使用情况动态调整Campaign容量
    """
    if current_usage < 0.5:  # 使用率低于50%
        new_tps = max(1, int(current_usage * 2))
        
        personalize.update_campaign(
            campaignArn=campaign_arn,
            minProvisionedTPS=new_tps
        )
```

## 10. 安全与合规

### 10.1 数据安全

1. **数据加密**
   - S3静态加密
   - 传输中加密

2. **访问控制**
   - IAM角色最小权限原则
   - VPC端点配置

### 10.2 合规要求

```python
# 数据脱敏示例
def anonymize_store_data(store_id: str) -> str:
    """
    对商店ID进行哈希处理
    """
    import hashlib
    return hashlib.sha256(store_id.encode()).hexdigest()[:16]
```

## 11. 部署清单

### 11.1 Prerequisites

- [ ] AWS账户及适当的IAM权限
- [ ] S3存储桶用于数据存储
- [ ] 历史采购数据已准备并清洗

### 11.2 部署步骤

1. **数据准备**
   ```bash
   # 1. 上传训练数据到S3
   aws s3 cp interactions.csv s3://your-bucket/training-data/
   
   # 2. 验证数据格式
   python validate_data.py
   ```

2. **创建Personalize资源**
   ```python
   # 按顺序执行以下脚本
   python create_dataset_group.py
   python create_schema.py
   python create_dataset.py
   python import_data.py
   python create_solution.py
   python create_campaign.py
   python create_filter.py
   ```

3. **部署API服务**
   ```bash
   # 部署Lambda函数
   sam deploy --template-file template.yaml
   
   # 配置API Gateway
   aws apigateway create-rest-api --name store-recommendations
   ```

4. **配置监控**
   ```bash
   # 设置CloudWatch告警
   aws cloudwatch put-metric-alarm --alarm-name "RecommendationErrors"
   ```

### 11.3 验证测试

```python
# 测试推荐API
def test_recommendation_api():
    import requests
    
    response = requests.get(
        'https://your-api-gateway-url/recommendations/store_001'
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'recommendations' in data
    assert len(data['recommendations']) > 0
```

## 12. 故障排除

### 12.1 常见问题

1. **推荐结果为空**
   - 检查过滤器配置
   - 验证商店ID是否存在于训练数据中
   - 确认Campaign状态为ACTIVE

2. **推荐质量差**
   - 增加训练数据量
   - 调整算法超参数
   - 检查数据质量

3. **API响应慢**
   - 增加Campaign的TPS配置
   - 优化Lambda函数代码
   - 使用缓存策略

### 12.2 调试工具

```python
def debug_recommendation_quality(store_id: str):
    """
    调试推荐质量
    """
    # 获取商店的历史采购
    historical_purchases = get_store_purchases(store_id)
    
    # 获取推荐结果
    recommendations = get_recommendations(store_id)
    
    # 分析推荐覆盖率
    total_products = get_total_product_count()
    purchased_products = len(historical_purchases)
    coverage = len(recommendations) / (total_products - purchased_products)
    
    print(f"Store {store_id} coverage: {coverage:.2%}")
    
    return {
        'historical_purchases': len(historical_purchases),
        'recommendations': len(recommendations),
        'coverage': coverage
    }
```

## 13. 结论

本设计文档详细阐述了如何使用AWS Personalize构建基于商品购买比率的新品推荐系统。主要特点包括：

1. **简化的数据模型**：仅使用商店ID、商品ID和采购历史
2. **智能过滤**：自动排除已购买商品
3. **实时推荐**：支持实时API调用
4. **可扩展架构**：支持批量处理和实时更新
5. **成本优化**：动态调整资源配置

通过遵循本文档的实施步骤，您可以构建一个高效、准确的商店新品推荐系统，帮助商店发现潜在的采购机会。