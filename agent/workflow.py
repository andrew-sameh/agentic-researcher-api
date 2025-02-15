import json
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from agent.calls_schema import DecisionMakingOutput, JudgeOutput
from agent.state import AgentState
from agent.tools import  search_papers, download_paper,ask_human_feedback
from agent.prompts import decision_making_prompt, planning_prompt, agent_prompt, judge_prompt
from utils.helpers import format_tools_description

class ResearchAgent:
    """Encapsulates LangGraph agent logic."""

    def __init__(self):
        self.base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.tools = [search_papers, download_paper, ask_human_feedback]
        self.tools_dict = { tool.name: tool for tool in self.tools}
        self.decision_making_llm = self.base_llm.with_structured_output(DecisionMakingOutput)
        self.agent_llm = self.base_llm.bind_tools(self.tools)
        self.judge_llm = self.base_llm.with_structured_output(JudgeOutput)
        self.workflow = self._build_workflow()
        self.compiled_agent = self.workflow.compile()

    def _build_workflow(self):
        """Creates and configures the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("decision_making", self.decision_making_node)
        workflow.add_node("planning", self.planning_node)
        workflow.add_node("tools", self.tools_node)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("judge", self.judge_node)

        workflow.set_entry_point("decision_making")
        workflow.add_conditional_edges("decision_making", self.router, {"planning": "planning", "end": END})
        workflow.add_edge("planning", "agent")
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges("agent", self.should_continue, {"continue": "tools", "end": "judge"})
        workflow.add_conditional_edges("judge", self.final_answer_router, {"planning": "planning", "end": END})

        return workflow

    # Decision making node
    def decision_making_node(self, state: AgentState):
        """Entry point of the workflow. Based on the user query, the model can either respond directly or perform a full research, routing the workflow to the planning node"""
        system_prompt = SystemMessage(content=decision_making_prompt)
        response: DecisionMakingOutput = self.decision_making_llm.invoke([system_prompt] + state["messages"])
        output = {"requires_research": response.requires_research}
        if response.answer:
            output["messages"] = [AIMessage(content=response.answer)]
        return output

    # Task router function
    def router(self, state: AgentState):
        """Router directing the user query to the appropriate branch of the workflow."""
        return "planning" if state["requires_research"] else "end"

    # Planning node
    def planning_node(self, state: AgentState):
        """Planning node that creates a step by step plan to answer the user query."""
        system_prompt = SystemMessage(content=planning_prompt.format(tools=format_tools_description(self.tools)))
        response = self.base_llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    # Tool call node
    def tools_node(self, state: AgentState):
        """Tool call node that executes the tools based on the plan."""
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = self.tools_dict[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    # Agent call node
    def agent_node(self, state: AgentState):
        """Agent call node that uses the LLM with tools to answer the user query."""
        system_prompt = SystemMessage(content=agent_prompt)
        response = self.agent_llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}
    # Should continue function
    def should_continue(self, state: AgentState):
        """Check if the agent should continue or end."""
        return "continue" if state["messages"][-1].tool_calls else "end"
    # Judge node
    def judge_node(self, state: AgentState):
        """Node to let the LLM judge the quality of its own final answer."""
        # End execution if the LLM failed to provide a good answer twice.
        num_feedback_requests = state.get("num_feedback_requests", 0)
        if num_feedback_requests >= 2:
            return {"is_good_answer": True}
        system_prompt = SystemMessage(content=judge_prompt)
        response: JudgeOutput = self.judge_llm.invoke([system_prompt] + state["messages"])
        output = {
            "is_good_answer": response.is_good_answer,
            "num_feedback_requests": num_feedback_requests + 1,
        }
        if response.feedback:
            output["messages"] = [AIMessage(content=response.feedback)]
        return output
    # Final answer router function
    def final_answer_router(state: AgentState):
        """Router to end the workflow or improve the answer."""
        return "end" if state["is_good_answer"] else "planning"
