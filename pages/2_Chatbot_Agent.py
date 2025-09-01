import streamlit as st
import utils

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import WikipediaQueryRun

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool


st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header("Chatbot with Web Browser Access")

st.write("Equipped with Tavily search agent, Wikipedia, and Arxiv tools.")

class ChatbotTools:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4o-mini"

    def setup_agent(self):
        # Check if Bing API key is available
        bing_subscription_key = os.getenv('BING_SUBSCRIPTION_KEY')
        wiki_agent = WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper())

        if bing_subscription_key:
            # Use BingSearch if API key is available
            bing_search = BingSearchAPIWrapper()
            tools = [
                Tool(
                    name="BingSearch",
                    func=bing_search.run,
                    description="Useful for when you need to answer questions about current events. You should ask targeted questions",
                ),
                Tool(
                name="Wikipedia",
                func=wiki_agent.run,
                description="Useful for when you need to query about a specific topic, person, or event. You should ask targeted questions",
            )
            ]
        else:
            # Fallback to DuckDuckGo if Bing API key is not available
            ddg_search = DuckDuckGoSearchRun()
            tools = [
                Tool(
                    name="DuckDuckGoSearch",
                    func=ddg_search.run,
                    description="Useful for when you need to answer questions about current events. You should ask targeted questions",
                )
            ]

        # Setup LLM and Agent
        llm = ChatOpenAI(model_name=self.openai_model, streaming=True)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )
        return agent
    @utils.enable_chat_history
    def main(self):
        agent = self.setup_agent()
        if not agent:
            return

        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, "user")
            with st.chat_message("assistant"):
                try:
                    placeholder = st.empty()
                    acc = ""

                    # Stream state updates (chunked by reasoning/tool steps)
                    for update in agent.stream({"messages": user_query}):
                        msgs = update.get("messages", [])
                        for m in msgs:
                            content = getattr(m, "content", "")
                            if not content and isinstance(getattr(m, "content", None), list):
                                content = "".join(
                                    c.get("text", "")
                                    for c in m.content
                                    if isinstance(c, dict) and c.get("type") == "text"
                                )
                            if content:
                                acc += content
                                placeholder.markdown(acc)

                    # Fallback if nothing streamed
                    if not acc:
                        resp = agent.invoke({"messages": user_query})
                        acc = (
                            resp["messages"][-1].content
                            if isinstance(resp, dict) and resp.get("messages")
                            else str(resp)
                        )
                        placeholder.markdown(acc)

                    st.session_state.messages.append({"role": "assistant", "content": acc})

                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()
