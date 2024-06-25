import streamlit as st


st.set_page_config(
    page_title="Judge AI",
    page_icon=":weight_lifter:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.judgeai.com/help",  # ejemplo de como sería una vez montado
        "Report a bug": "https://www.judgeai.com/bug",  # ejemplo de como sería una vez montado
        "About": "# This is JudgeAi. This is our Data Science Final Proyect !",
    },
)


# def main_page():
#     st.markdown("# 🏠Home")
#     st.sidebar.markdown("# 🏠Home")


# def page2():
#     st.markdown("# 🕺Profile")
#     st.sidebar.markdown("#🕺Profile")


# def page3():
#     st.markdown("# ⚖️Judge AI App")
#     st.sidebar.markdown("#⚖️Judge AI App")


# def page4():
#     st.markdown("# 🕵️‍♂️About us")
#     st.sidebar.markdown("# 🕵️‍♂️About us")


# page_names_to_funcs = {
#     "🏠Home": main_page,
#     "🕺Profile": page2,
#     "⚖️Judge AI App": page3,
#     "🕵️‍♂️About us": page4,
# }

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()
