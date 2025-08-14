from graph import create_graph
from langchain_core.messages import HumanMessage

agent_app = create_graph()

config = {
    "configurable": {
        "thread_id": 1,
        "recursion_limit": 50
    }
}

while True:
    user_prompt = input("Enter your query: ")
    if user_prompt.lower() == "exit":
        break
    
    initial_state = {
        "messages": [HumanMessage(content=user_prompt)]
    }
    
    response = agent_app.invoke(
        initial_state,
        config=config
    )
    
    # The response is the final state, which contains a 'messages' list
    # Get the last message (which should be the AI's response)
    if response.get("messages") and len(response["messages"]) > 1:
        ai_response = response["messages"][-1].content
        print(f"\n\nAI response: {ai_response}\n\n")
    else:
        print("\n\nNo response generated.\n\n")