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
#     st.markdown("# 🕵️‍♂️About_us")
#     st.sidebar.markdown("# 🕵️‍♂️About_us")


# page_names_to_funcs = {
#     "🏠Home": main_page,
#     "🕺Profile": page2,
#     "⚖️Judge AI App": page3,
#     "🕵️‍♂️About_us": page4,
# }

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()


# name = st.text_input(label="Name", max_chars=20, placeholder="Tu nombre")


# password = st.text_input(
#     label="Contraseña", placeholder="Tu contraseña", type="password"
# )

# st.title(name)
# st.title(password)

# # Text Area
# texto = st.text_area(
#     label="Enter Text", height=150, max_chars=2000, placeholder="Review"
# )
# st.write(texto)

# # Input Numbers
# number = st.number_input(
#     label="Enter Number", min_value=-256, max_value=255, value=0, step=10
# )

# # Date Input
# fecha = st.date_input(label="Tutoria")

# # Time Input
# tiempo = st.time_input(label="Hora")

# # Color Picker
# color = st.color_picker("Select Color")
# st.write(f"Your color: {color}")
