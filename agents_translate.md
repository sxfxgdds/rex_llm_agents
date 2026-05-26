# 项目：REx —— 基于声誉调节的LLM智能体探索机制

## 目标

构建一个最小但可扩展的Python研究原型，用于研究LLM智能体进化博弈中基于声誉调节的探索行为。

系统应模拟多个LLM智能体在重复社会困境博弈中的行为。每个智能体拥有一个声誉分数，声誉动态调节智能体的探索行为，包括采样温度、反思深度和策略风险倾向。

## 严格范围

- 仅实现LLM智能体。
- 不实现强化学习（RL）智能体、多智能体强化学习（MARL）智能体或基于主体建模（ABM）风格的智能体作为核心。
- 不训练模型参数。
- 不使用梯度更新。
- 这是一个无梯度的多智能体仿真框架。
- 默认后端必须是模拟LLM后端，使整个项目无需外部API调用即可运行。
- 之后可以添加兼容OpenAI的真实后端，但不能作为默认使用。

## 编码规范

- Python 3.10+。
- 使用类型提示。
- 适当时使用dataclass。
- 保持模块小而聚焦。
- 避免将所有逻辑放在一个文件中。
- 每个随机过程必须可设定种子、可复现。
- 优先使用简单、可读的研究代码，而非过度设计的抽象。
- 在默认测试或基线运行中不调用外部API。
- 使用CSV记录智能体级别和回合级别的结果。
- 包含仅使用matplotlib的绘图脚本。
- 避免使用seaborn。
- 每张图应为独立图形，不使用子图。

## 必须可运行的命令

以下命令必须能正常执行：

```bash
python scripts/run_baseline.py
python scripts/run_reputation_attack.py
python scripts/run_malicious_agents.py
python scripts/plot_results.py
pytest
```

## 实现优先级

首先实现：
- MockBackend（模拟后端）
- 囚徒困境
- ReputationModel（声誉模型）
- ReputationExplorationController（声誉探索控制器）
- LLMAgent（LLM智能体）
- SimulationRunner（仿真运行器）
- 基线实验
- CSV日志记录
- 绘图

然后实现：
- ReputationAttack（声誉攻击）
- MaliciousAgentInjection（恶意智能体注入）
- 恢复指标
- 测试

可选的后续扩展：
- TrustGame（信任博弈）
- PublicGoodsGame（公共品博弈）
- 兼容OpenAI的后端

## 研究逻辑

主要的实验对比应包括：

1. fixed_low_temperature（固定低温）
2. fixed_mid_temperature（固定中温）
3. fixed_high_temperature（固定高温）
4. reputation_linear（线性声誉）
5. reputation_piecewise（分段声誉）
6. reputation_piecewise_with_attack（带攻击的分段声誉）
7. reputation_piecewise_with_malicious_agents（带恶意智能体的分段声誉）

最重要的方法是 reputation_piecewise（分段声誉）：

```python
If reputation < 0.3:
    temperature = 0.2
    reflection_depth = 3
    risk_tendency = "low"
    mode = "recovery"

If 0.3 <= reputation < 0.7:
    temperature = 0.5
    reflection_depth = 2
    risk_tendency = "medium"
    mode = "normal"

If reputation >= 0.7:
    temperature = 0.75
    reflection_depth = 1
    risk_tendency = "medium_high"
    mode = "exploration"
```

还需实现声誉冲击响应：
如果智能体在最近3轮内发生过背叛，则暂时将其温度乘以0.6进行降低。

## 验证

每个主要实现步骤完成后，运行相关命令并修复错误后再继续。
不在核心逻辑中留下TODO占位符。
完成后，总结：
- 创建的文件
- 运行的命令
- 测试通过/失败情况
- 已知限制
