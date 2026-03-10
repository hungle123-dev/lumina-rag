from langchain_openai import ChatOpenAI
from typing import Dict, List
from langchain_core.documents import Document
from config.settings import settings

class ResearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.RESEARCH_MODEL,
            temperature=0,
        )
        print(f"OpenRouter initialized successfully for ResearchAgent (model: {settings.RESEARCH_MODEL}).")
    
    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - You are a precise Question-Answering system. 
        - Your PRIMARY goal is to answer the **Question** below using ONLY the provided context.
        - If the context does not contain the answer, state that you cannot answer it.
        - DO NOT provide a general summary of the documents unless specifically asked for a summary.
        - Focus strictly on the data, values, and facts requested.
        - Be clear, concise, and factual.
        
        **Question:** {question}
        **Context:**
        {context}
        
        **Provide your precise answer below (no generic summaries unless requested):**
        """
        return prompt
    
    def generate(self, question: str, documents: List[Document]) -> Dict:
        print(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context)
        print("Prompt created for the LLM.")

        try:
            response = self.llm.invoke(prompt)
            print("LLM response received.")
            llm_response = response.content
            print(f"Raw LLM response:\n{llm_response}")
        except Exception as e:
            print(f"Error during model inference: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }