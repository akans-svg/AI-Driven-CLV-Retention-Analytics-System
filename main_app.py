import streamlit as st
import json
import os

# ---------- USER DATABASE FILE ----------
USER_DB = "users.json"

# ---------- CREATE USER FILE IF NOT EXISTS ----------
if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

# ---------- LOAD USERS ----------
def load_users():
    with open(USER_DB, "r") as f:
        return json.load(f)

# ---------- SAVE USERS ----------
def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

# ---------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------- LOGIN PAGE ----------
def login_page():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()

        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ---------- REGISTER PAGE ----------
def register_page():
    st.title("Register")

    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    if st.button("Register"):
        users = load_users()

        if username in users:
            st.error("User already exists")
        else:
            users[username] = password
            save_users(users)
            st.success("Registration successful! Please login.")

# ---------- DASHBOARD ----------
def dashboard():
    st.sidebar.success("Logged in successfully")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("AI-Driven CLV & Retention Analytics System")

    # Import your dashboard
    import streamlit_app

# ---------- MAIN APP ----------
def main():

    if st.session_state.logged_in:
        dashboard()

    else:
        page = st.sidebar.selectbox(
            "Select Page",
            ["Login", "Register"]
        )

        if page == "Login":
            login_page()
        else:
            register_page()

main()