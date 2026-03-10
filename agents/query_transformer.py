from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class QueryTransformer:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.RESEARCH_MODEL,
            temperature=0.3, # Slightly creative to generate new keywords
            max_retries=2
        )
        
        self.prompt = PromptTemplate(
            input_variables=["original_question", "verification_report"],
            template="""You are an expert search query optimization assistant.
Your task is to rewrite a user's question to be more effective for a semantic search engine, based on why the previous search failed.

Original User Question:
{original_question}

Verification Report (Why the previous answer failed):
{verification_report}

Instructions:
1. Analyze the Verification Report to understand what information was missing or contradicted.
2. Rewrite the Original User Question to specifically target the missing or problematic information.
3. Make the new query concise, keyword-rich, and optimized for retrieval.
4. Output ONLY the rewritten query text, nothing else.

Optimized Query:"""
        )

        self.chain = self.prompt | self.llm

    def rewrite(self, original_question: str, verification_report: str) -> str:
        logger.info(f"Rewriting query for: {original_question}")
        try:
            response = self.chain.invoke({
                "original_question": original_question,
                "verification_report": verification_report
            })
            new_query = response.content.strip()
            logger.info(f"Rewritten query: {new_query}")
            return new_query
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return original_question # Fallback to original if rewriting fails
