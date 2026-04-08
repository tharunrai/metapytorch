import uvicorn
from server.environment import app

def main():
    """Entry point for the multi-mode deployment"""
    # The validator requires uvicorn to run inside this main function
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()