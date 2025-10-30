# sales_assistant.py
"""
上下文感知智能销售助手 (Context-Aware Sales Assistant)
使用 LangChain + LLM + RAG 构建
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_community.chat_models import ChatZhipuAI
# LangChain 相关组件
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
# from langchain_community.llms import Ollama  # 使用本地 Ollama 模型，也可替换为其他 LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CustomerContext:
    """客户上下文信息"""
    name: str
    company: str
    industry: str
    position: str
    interests: List[str]
    pain_points: List[str]
    previous_interactions: List[str]
    budget_range: str
    decision_timeline: str


@dataclass
class SalesContext:
    """销售场景上下文"""
    meeting_type: str  # "initial", "follow_up", "demo", "negotiation"
    current_stage: str
    goals: List[str]
    constraints: List[str]
    customer_emotion: str  # "interested", "skeptical", "urgent", "neutral"


class KnowledgeBase:
    """产品知识库管理"""
    
    def __init__(self, data_dir: str = "data/knowledge_base"):
        self.data_dir = data_dir
        self.embeddings = ZhipuAIEmbeddings(model="embedding-3",
                                            api_key="0f4ae0b90dff44389836ecf634297560.c1eOL2jdMckW1bfO")
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_documents(self):
        """加载知识库文档"""
        # documents = TextLoader("data/knowledge_base/835033325.txt", autodetect_encoding=True).load()
        # print(documents)
        # logger.info(f"加载了 {len(documents)} 个文档")
        # return documents
        try:
            loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'autodetect_encoding': True}
            )
            documents = loader.load()
            # documents = TextLoader("data/knowledge_base/835033325.txt").load()
            # print(documents)
            logger.info(f"加载了 {len(documents)} 个文档")
            return documents
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            return []
    
    def create_vectorstore(self):
        """创建向量数据库"""
        documents = self.load_documents()
        if not documents:
            logger.warning("没有找到文档，创建空的知识库")
            return
        
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"分割成 {len(texts)} 个文本块")
        
        # 创建向量数据库
        self.vectorstore = Chroma.from_documents(
            texts,
            self.embeddings,
            persist_directory="./chroma_db"
        )
        logger.info("向量数据库创建完成")
    
    def get_retriever(self):
        
        if self.vectorstore is None:
            self.create_vectorstore()
        
        if self.vectorstore is None:
            raise ValueError("向量数据库未创建，请检查知识库文件是否加载成功")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})


class SalesAssistant:
    """智能销售助手主类"""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.llm = ChatZhipuAI(
            temperature=0.95,
            model="glm-4",
            api_key="0f4ae0b90dff44389836ecf634297560.c1eOL2jdMckW1bfO",
        )
        self.retriever = self.knowledge_base.get_retriever()
        
        # 创建 RAG 链
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # 自定义提示模板
        self.prompt_template = """
        你是一位专业的销售顾问，正在帮助销售代表与客户 {customer_name} 进行沟通。
        客户公司：{company}
        行业：{industry}
        职位：{position}
        关注点：{interests}
        痛点：{pain_points}
        预算：{budget_range}
        决策时间：{decision_timeline}

        当前销售场景：
        - 会议类型：{meeting_type}
        - 销售阶段：{current_stage}
        - 目标：{goals_str}
        - 限制：{constraints_str}
        - 客户情绪：{customer_emotion}

        相关知识库信息：
        {context}

        请根据以上信息，提供专业的销售建议：
        1. 针对客户痛点的解决方案建议
        2. 产品优势的针对性介绍
        3. 应对客户可能提出的问题
        4. 推进销售进程的策略
        5. 建议的沟通话术

        请用专业、简洁、有说服力的语言回答。
        """
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=[
                "customer_name", "company", "industry", "position",
                "interests", "pain_points", "budget_range", "decision_timeline",
                "meeting_type", "current_stage", "goals_str", "constraints_str",
                "customer_emotion", "context"
            ]
        )
    
    def _format_context(self, source_documents) -> str:
        """格式化检索到的上下文"""
        context = ""
        for i, doc in enumerate(source_documents):
            context += f"【参考信息 {i + 1}】\n{doc.page_content}\n来源: {doc.metadata.get('source', '未知')}\n\n"
        return context
    
    def _prepare_context_input(self, customer_context: CustomerContext,
                               sales_context: SalesContext) -> Dict:
        """准备上下文输入"""
        return {
            "customer_name": customer_context.name,
            "company": customer_context.company,
            "industry": customer_context.industry,
            "position": customer_context.position,
            "interests": ", ".join(customer_context.interests),
            "pain_points": ", ".join(customer_context.pain_points),
            "budget_range": customer_context.budget_range,
            "decision_timeline": customer_context.decision_timeline,
            "meeting_type": sales_context.meeting_type,
            "current_stage": sales_context.current_stage,
            "goals_str": ", ".join(sales_context.goals),
            "constraints_str": ", ".join(sales_context.constraints),
            "customer_emotion": sales_context.customer_emotion
        }
    
    def get_sales_advice(self,
                         customer_context: CustomerContext,
                         sales_context: SalesContext) -> Dict:
        """
        获取销售建议

        Args:
            customer_context: 客户上下文
            sales_context: 销售场景上下文

        Returns:
            包含建议和来源的字典
        """
        try:
            # 检索相关知识
            query = f"解决{customer_context.industry}行业客户关于{', '.join(customer_context.pain_points)}的问题"
            retrieval_result = self.retriever.invoke(query)
            
            # 准备输入
            context_input = self._prepare_context_input(customer_context, sales_context)
            context_input["context"] = self._format_context(retrieval_result)
            
            # 创建最终提示
            final_prompt = self.prompt.format(**context_input)
            
            # 调用 LLM
            response = self.llm.invoke(final_prompt)
            
            return {
                "advice": response,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "未知")
                    }
                    for doc in retrieval_result
                ],
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"获取销售建议失败: {e}")
            return {
                "advice": f"抱歉，获取销售建议时出现错误：{str(e)}",
                "source_documents": [],
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def analyze_customer_sentiment(self, conversation_text: str) -> Dict:
        """
        分析客户情绪（简单实现）
        """
        prompt = f"""
        请分析以下客户对话的情绪状态，从以下类别中选择最合适的：
        - 兴奋 (Excited)
        - 感兴趣 (Interested)
        - 怀疑 (Skeptical)
        - 犹豫 (Hesitant)
        - 紧迫 (Urgent)
        - 中立 (Neutral)
        - 不满 (Dissatisfied)

        对话内容：
        {conversation_text}

        请只返回情绪类别，并简要说明理由。
        """
        
        try:
            response = self.llm.invoke(prompt)
            return {"sentiment": response.strip(), "success": True}
        except Exception as e:
            return {"sentiment": "分析失败", "error": str(e), "success": False}


# 示例使用和测试
def create_sample_data():
    """创建示例数据"""
    os.makedirs("data/knowledge_base", exist_ok=True)
    
    # 创建产品知识库示例
    knowledge_content = """
    产品名称：智能CRM系统
    核心功能：
    1. 客户关系管理：全面跟踪客户信息和互动历史
    2. 销售管道自动化：自动推进销售流程，减少手动操作
    3. 数据分析仪表板：实时销售数据可视化
    4. 智能提醒：关键节点自动提醒
    5. 移动端支持：随时随地访问客户信息

    行业解决方案：
    - 制造业：设备维护提醒，供应链协同
    - 零售业：客户行为分析，个性化推荐
    - 金融业：合规性管理，风险评估
    - 教育行业：学员管理，课程推荐

    定价方案：
    基础版：$99/用户/月，包含核心CRM功能
    专业版：$199/用户/月，增加自动化和分析功能
    企业版：定制报价，包含API集成和高级安全

    常见客户痛点解决方案：
    - 客户跟进不及时：通过自动化提醒解决
    - 销售预测不准：通过AI预测模型解决
    - 数据分散：通过统一平台整合
    - 团队协作困难：通过共享仪表板解决
    """
    
    with open("data/knowledge_base/product_info.txt", "w", encoding="utf-8") as f:
        f.write(knowledge_content)


def main():
    """主函数 - 演示使用"""
    print("🚀 启动上下文感知智能销售助手...")
    
    # 创建示例数据
    create_sample_data()
    
    # 初始化助手
    assistant = SalesAssistant()
    
    # 创建客户上下文
    customer = CustomerContext(
        name="张伟",
        company="创新科技有限公司",
        industry="制造业",
        position="销售总监",
        interests=["销售自动化", "数据分析", "团队协作"],
        pain_points=["客户跟进不及时", "销售预测不准"],
        previous_interactions=[
            "2025-10-25: 初步电话沟通，表达了对自动化功能的兴趣",
            "2025-10-28: 发送了产品资料和案例研究"
        ],
        budget_range="中等",
        decision_timeline="1-2个月"
    )
    
    # 创建销售场景上下文
    sales_context = SalesContext(
        meeting_type="follow_up",
        current_stage="需求确认",
        goals=["确认具体需求", "安排产品演示", "了解预算细节"],
        constraints=["客户时间紧张", "需要高层批准"],
        customer_emotion="interested"
    )
    
    print("\n👤 客户信息:")
    print(f"  名称: {customer.name}")
    print(f"  公司: {customer.company}")
    print(f"  行业: {customer.industry}")
    print(f"  痛点: {', '.join(customer.pain_points)}")
    
    print(f"\n🎯 销售场景:")
    print(f"  会议类型: {sales_context.meeting_type}")
    print(f"  当前阶段: {sales_context.current_stage}")
    print(f"  目标: {', '.join(sales_context.goals)}")
    
    print("\n💡 正在获取销售建议...")
    
    # 获取销售建议
    result = assistant.get_sales_advice(customer, sales_context)
    
    if result["success"]:
        print("\n✅ 智能销售建议:")
        print(result["advice"])
        
        print(f"\n📚 信息来源:")
        for i, source in enumerate(result["source_documents"]):
            print(f"  [{i + 1}] {source['source']}")
    else:
        print(f"❌ 错误: {result['advice']}")
    
    # 情绪分析示例
    conversation = """
    客户：我对你们的自动化功能很感兴趣，特别是客户跟进提醒。
    但是价格方面还需要和财务部门讨论。
    我们确实需要解决跟进不及时的问题，这影响了我们的成交率。
    我很高兴你们能提供这方面的解决方案。
    """
    
    sentiment = assistant.analyze_customer_sentiment(conversation)
    print(f"\n😊 客户情绪分析: {sentiment.get('sentiment', '未知')}")


if __name__ == "__main__":
    main()
