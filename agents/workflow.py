from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker
from .query_transformer import QueryTransformer
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    current_query: str  # Tracks the rewritten query
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever
    iteration_count: int # To prevent infinite loops

class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.transformer = QueryTransformer()
        self.max_iterations = 3
        self.compiled_workflow = self.build_workflow()

    def _check_relevance_step(self, state: AgentState) -> Dict:
        retriever = state["retriever"]
        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=retriever,
            k=20
        )

        if classification in ["CAN_ANSWER", "PARTIAL"]:
            return {"is_relevant": True}
        else:  # classification == "NO_MATCH"
            return {
                "is_relevant": False,
                "draft_answer": "This question isn't related (or there's no data) for your query. Please ask another question relevant to the uploaded document(s)."
            }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def _research_step(self, state: AgentState) -> Dict:
        query = state.get("current_query", state["question"])
        logger.info(f"Research step using query: '{query}' (Iteration {state.get('iteration_count', 1)})")
        
        # We re-retrieve documents if the query changed
        if state.get("iteration_count", 1) > 1:
             docs = state["retriever"].invoke(query)
        else:
             docs = state["documents"]

        result = self.researcher.generate(query, docs)
        return {
            "draft_answer": result["draft_answer"],
            "documents": docs # Update documents in state
        }
    
    def _verification_step(self, state: AgentState) -> Dict:
        logger.info("Verifying the draft answer...")
        result = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": result["verification_report"]}
    
    def _decide_next_step(self, state: AgentState) -> str:
        verification_report = state["verification_report"]
        iteration_count = state.get("iteration_count", 1)

        if ("Supported: NO" in verification_report or "Relevant: NO" in verification_report):
            if iteration_count < self.max_iterations:
                logger.info(f"Verification failed. Transforming query (Iteration {iteration_count}/{self.max_iterations})...")
                return "transform_query"
            else:
                logger.warning("Max iterations reached. Ending workflow despite verification failure.")
                return "end"
        else:
            logger.info("Verification successful, ending workflow.")
            return "end"

    def _transform_query_step(self, state: AgentState) -> Dict:
        current_query = state.get("current_query", state["question"])
        new_query = self.transformer.rewrite(
            original_question=current_query,
            verification_report=state["verification_report"]
        )
        
        return {
            "current_query": new_query,
            "iteration_count": state.get("iteration_count", 1) + 1
        }

    def build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)
        workflow.add_node("transform_query", self._transform_query_step)

        workflow.set_entry_point("check_relevance")
        
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END
            }
        )
        
        workflow.add_edge("research", "verify")
        
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "transform_query": "transform_query",
                "end": END
            }
        )
        
        workflow.add_edge("transform_query", "research")
        
        return workflow.compile()

    def full_pipeline(self, question: str, retriever: EnsembleRetriever):
        try:
            logger.info(f"Starting pipeline with question: '{question}'")
            documents = retriever.invoke(question)
            
            initial_state = AgentState(
                question=question,
                current_query=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever,
                iteration_count=1
            )
            
            final_state = self.compiled_workflow.invoke(initial_state)
            
            # If verification failed on the final try, mention it in the answer
            final_answer = final_state["draft_answer"]
            if final_state.get("iteration_count", 1) >= self.max_iterations and "Supported: NO" in final_state["verification_report"]:
                final_answer += "\n\n*(Note: I am not fully confident in this answer based on the provided documents. I retried multiple times but could not find exact supporting evidence.)*"

            return {
                "draft_answer": final_answer,
                "verification_report": final_state["verification_report"],
                "final_query_used": final_state.get("current_query", question)
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise