import streamlit as st
from news_summarizer import NewsArticleSummarizer
from dotenv import load_dotenv
import os


load_dotenv()

def main():
    st.set_page_config(
        page_title="News Summarizer",
        page_icon="🤖",
        layout="centered"
    )

    # initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "llm_app" not in st.session_state:
        st.session_state.llm_app = None

    # Title and description
    st.title("News Summarizer")
    if st.session_state.get("summary_result") is not None:
        result = st.session_state["summary_result"]
        st.markdown(f"Title: {result["title"]}")
        st.markdown(f"Published Date: {result["publish_date"]}")
        st.markdown(result["summary"]["input_documents"][0].page_content)
    else:
        st.markdown("Enter a news URL to get the summary of the news")

    # Implement sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Openai API key input
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")

        # Model Selection
        model = st.selectbox(
            "Model",
            [
                "llama3.1:8b",
                "llama-3.3-70b",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-120b",
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano"
            ],
            help="Select the model to use"
        )

        # News Article URL
        news_article_url = st.sidebar.text_input("News Article URL", type="default", help="Enter your News Article URL")

        # Process News Article
        if st.button("Process News Article", use_container_width=True):
            st.session_state.messages = []
            st.session_state.llm_app = NewsArticleSummarizer(model, api_key=openai_api_key)
            result = st.session_state.llm_app.summarize(news_article_url)
            st.session_state['summary_result'] = result
            st.rerun()


    if st.session_state.llm_app is None:
        try:
            st.session_state.llm_app = NewsArticleSummarizer(model, api_key=openai_api_key)
        except Exception as e:
            st.error(f"Error initializing News Summarizer App: {str(e)}")


if __name__ == "__main__":
    main()