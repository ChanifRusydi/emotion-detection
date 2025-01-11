import os
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Connect to MongoDB
db_pass = os.getenv("MONGODB_PASS")
uri = f"mongodb+srv://user1:{db_pass}@cluster0.kexyy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)

try:
    client.admin.command("ping")
    print("Connected to MongoDB")
except Exception as e:
    print("Unable to connect to MongoDB")
    print(e)
    print("Will exit now.")
    os._exit(1)

st.title("Dashboard for Video Emotion Detection")
st.header("Users")


