import streamlit as st

from agent import run_agent

st.title("Company Risk Assistant")
st.caption("Built by Shraddha Nimbalkar")
st.caption("Ask about a company number, for example: How risky is company 10? "
           "A human analyst makes the final decision.")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, text, steps in st.session_state.chat:
    with st.chat_message(role):
        st.write(text)
        if steps:
            with st.expander("Steps taken"):
                for s in steps:
                    st.code(s)

question = st.chat_input("Ask a question")
if question:
    st.session_state.chat.append(("user", question, []))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, steps = run_agent(question)
        st.write(answer)
        if steps:
            with st.expander("Steps taken"):
                for s in steps:
                    st.code(s)
    st.session_state.chat.append(("assistant", answer, steps))
