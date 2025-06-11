import streamlit as st
st.title("My first To-Do List App")

task = st.text_input("Enter a new task")

if 'tasks' not in st.session_state:
    st.session_state.tasks = []

if st.button ("Add Task"):
    st.session_state.tasks.append(task)
    st.success(f"Added: {task}")

st.write("## Current Tasks")
for i, t in enumerate(st.session_state.tasks, 1):
    st.write(f"{i}.{t}")
