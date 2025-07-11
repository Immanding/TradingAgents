from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# 创建自定义配置
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "google"  # 使用不同的模型
config["backend_url"] = "https://generativelanguage.googleapis.com/v1"  # 使用不同的后端
config["deep_think_llm"] = "gemini-2.0-flash"  # 使用不同的模型
config["quick_think_llm"] = "gemini-2.0-flash"  # 使用不同的模型
config["max_debate_rounds"] = 1  # 增加辩论轮数
config["online_tools"] = True  # 增加辩论轮数

# 使用自定义配置初始化
ta = TradingAgentsGraph(debug=True, config=config)

# 前向传播
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# 记忆错误并反思
# ta.reflect_and_remember(1000) # 参数是持仓收益
