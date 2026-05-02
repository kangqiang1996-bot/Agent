import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------- 基础 Agent 类 ---------------------
class BaseAgent:
    """所有 Agent 的基类"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """统一的 LLM 调用接口"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 可换成 gpt-3.5-turbo 降低成本
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{self.name}] LLM 调用失败: {e}")
            return ""

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """子类需要实现的具体逻辑"""
        raise NotImplementedError

# --------------------- 具体 Agent 实现 ---------------------
class TrendAnalyzerAgent(BaseAgent):
    """趋势分析 Agent - 包含长链推理（思维链）"""
    def __init__(self):
        super().__init__("趋势分析师", "识别热点、分析原因、预测走向")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        topic = context.get("topic", "科技产品")
        system_prompt = """你是一名资深的社交媒体趋势分析师。你的回答必须包含清晰的推理步骤，格式如下：
        
【推理过程】
1. 当前用户讨论的核心焦点是什么？
2. 产生这一焦点的深层原因（技术/经济/文化）？
3. 未来 1~2 周的趋势演化方向（上升/下降/变异）？

【结论】给出三个关键洞察，每条洞察需包含具体数据和标签示例。"""

        user_prompt = f"""请针对运营主题「{topic}」，进行深度趋势分析。要求使用思维链逐步推理。"""

        result = self.call_llm(system_prompt, user_prompt, temperature=0.6)
        return {"trend_report": result}

class ContentPlannerAgent(BaseAgent):
    """内容策划 Agent - 基于趋势报告生成选题"""
    def __init__(self):
        super().__init__("内容策划师", "将趋势转化为可执行的选题")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trend_report = context.get("trend_report", "")
        system_prompt = """你是内容策划专家。基于趋势报告，生成 3 个不同的内容方向。
每个方向包含：
- 标题（吸引人、包含关键词）
- 核心角度（从哪个切入点讲故事）
- 目标受众（具体人群画像）
- 预期效果（例如引发讨论、种草、科普）"""

        user_prompt = f"""趋势报告如下：
{trend_report}

请策划 3 个内容方向，以 JSON 数组格式输出（只输出 JSON，不要其他解释）。"""

        raw = self.call_llm(system_prompt, user_prompt, temperature=0.8)
        # 尝试解析 JSON
        try:
            plans = json.loads(raw)
        except:
            # 降级处理：手动提取或直接当作字符串
            plans = [{"title": "解析失败", "raw": raw}]
        return {"content_plans": plans}

class CopywriterAgent(BaseAgent):
    """文案撰写 Agent - 根据每个选题生成实际文案"""
    def __init__(self):
        super().__init__("文案写手", "撰写适应多平台的运营文案")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        plans = context.get("content_plans", [])
        if not isinstance(plans, list):
            plans = []
        all_copy = []
        for idx, plan in enumerate(plans):
            if not plan:
                continue
            title = plan.get("title", "无标题")
            angle = plan.get("核心角度", plan.get("angle", ""))
            system_prompt = """你是小红书/抖音风格文案专家。为每个主题生成 2 个不同调性的短文案（每个 150 字以内）：
- 版本 A：情绪共鸣 + 痛点激发
- 版本 B：硬核干货 + 数据支撑
每个版本结尾带 3-5 个相关话题标签。"""
            user_prompt = f"""主题标题：{title}
核心角度：{angle}
请生成两个版本的文案（用分隔线 ---- 分开）。"""
            copies = self.call_llm(system_prompt, user_prompt, temperature=0.9)
            all_copy.append({
                "title": title,
                "angle": angle,
                "copy_texts": copies
            })
        return {"generated_copies": all_copy}

class ReviewerAgent(BaseAgent):
    """审核 Agent - 对文案评分并选出最佳"""
    def __init__(self):
        super().__init__("运营审核官", "评估质量、给出修改建议、最终决策")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        copies = context.get("generated_copies", [])
        if not copies:
            return {"final_decision": "无有效文案，请检查上游 Agent"}
        
        # 构建待审核的文案列表
        review_content = json.dumps(copies, ensure_ascii=False, indent=2)
        system_prompt = """你是严格的运营总监。从三个维度评分（1-10分）：吸引力、清晰度、合规性。
最终输出两条内容：
1. 综合评分最高的 1 个完整文案（注明来自哪个标题）
2. 给整个团队的 3 条具体改进建议
要求格式清晰，便于运营人员直接执行。"""
        user_prompt = f"""请审核以下所有文案候选集：
{review_content}
请给出最终决策和改进建议。"""
        decision = self.call_llm(system_prompt, user_prompt, temperature=0.4)
        return {"final_decision": decision}

# --------------------- 协调器（Orchestrator） ---------------------
class Orchestrator:
    """负责串联所有 Agent，管理上下文"""
    def __init__(self):
        self.agents = [
            TrendAnalyzerAgent(),
            ContentPlannerAgent(),
            CopywriterAgent(),
            ReviewerAgent()
        ]

    def run(self, initial_topic: str) -> Dict[str, Any]:
        print(f"\n🚀 开始多 Agent 协同运营自动化，主题：「{initial_topic}」\n" + "="*60)
        context = {"topic": initial_topic}
        
        for agent in self.agents:
            print(f"\n▶ 正在运行 Agent: {agent.name} ({agent.role})")
            result = agent.run(context)
            # 将结果合并到全局 context
            context.update(result)
            # 打印简要结果预览
            preview = str(result)[:200].replace("\n", " ")
            print(f"   ✓ {agent.name} 完成，产出预览: {preview}...")
        
        print("\n" + "="*60 + "\n✅ 全流程执行完毕！最终运营方案如下：\n")
        return context

# --------------------- 主程序入口 ---------------------
def main():
    # 可接受外部输入或使用默认主题
    topic = input("请输入本次运营的核心主题（例如：智能电动汽车、可持续时尚、AI办公工具）: ").strip()
    if not topic:
        topic = "可持续生活方式与零浪费消费"
        print(f"未输入，使用默认主题：{topic}")
    
    orchestrator = Orchestrator()
    final_context = orchestrator.run(topic)
    
    # 输出完整的运营报告
    print("\n📋 【最终运营报告】\n" + "-"*50)
    print("📈 趋势分析报告摘要：")
    print(final_context.get("trend_report", "无")[:500] + "...\n")
    
    print("📝 生成文案与审核决策：")
    print(final_context.get("final_decision", "无"))
    
    # 可选：保存到文件
    with open("operational_report.txt", "w", encoding="utf-8") as f:
        f.write(f"主题：{topic}\n")
        f.write("趋势分析报告：\n")
        f.write(final_context.get("trend_report", ""))
        f.write("\n\n审核最终决策：\n")
        f.write(final_context.get("final_decision", ""))
    print("\n💾 完整报告已保存到 operational_report.txt")

if __name__ == "__main__":
    main()
