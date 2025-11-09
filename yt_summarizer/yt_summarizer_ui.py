import streamlit as st
from youtube_video_summarizer import YoutubeVideoSummarizer
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    st.set_page_config(
        page_title="Youtube Video Summarizer",
        page_icon="🤖",
        layout="centered"
    )

    # initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "llm_app" not in st.session_state:
        st.session_state.llm_app = None

    # Title and description
    st.title("Youtube Video Summarizer")
    if st.session_state.get("video_result") is not None:
        result = st.session_state["video_result"]
        st.markdown(result["summary"]["output_text"])
    else:
        st.markdown("Enter A Youtube Video URL to get the summary of the video and query the transcript")

    # Implement sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Groq API key input
        groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")
        if not groq_api_key:
            groq_api_key = os.environ.get("GROQ_API_KEY")

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

        # Embedding Selection
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "openai",
                "chroma",
                "nomic",
            ],
            help="Select the embedding model to use"
        )

        # Youtube Video URL
        youtube_video_url = st.sidebar.text_input("Youtube Video URL", type="default", help="Enter your Youtube Video URL")

        # Process YouTube Video
        if st.button("Process Youtube Video", use_container_width=True):
            st.session_state.messages = []
            st.session_state.llm_app = YoutubeVideoSummarizer(model, embedding_model)
            result = st.session_state.llm_app.process_video(youtube_video_url)
            st.session_state['video_result'] = result
            st.rerun()

        # Clear chat button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            # if st.session_state.llm_app:
            #     st.session_state.llm_app.clear_history()
            st.rerun()

    if st.session_state.llm_app is None:
        try:
            st.session_state.llm_app = YoutubeVideoSummarizer(model, embedding_model)
        except Exception as e:
            st.error(f"Error initializing Youtube Video App: {str(e)}")

    # display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("You can now ask questions about the video here..."):
        if not openai_api_key:
            st.warning("Please enter your OPENAI API key in the sidebar")
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar")

        else:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            # get assistant's response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    try:

                        response = st.session_state['video_result']["qa_chain"].invoke({"question": prompt})
                        st.markdown(response['answer'])
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"{response['answer']}"
                            }
                        )
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()