from vertexai.preview.language_models import ChatModel, InputOutputTextPair #pip install google-cloud-aiplatform

response_text = ""  # 全域變數
def science_tutoring(message, initial_context, temperature: float = 0.2): #-> None
    global response_text  # 使用後續添加的 response_text 全域變數
    chat_model = ChatModel.from_pretrained("chat-bison@001")

    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 1024,    # Token limit determines the maximum amount of text output. default = 256
        "top_p": 0.95,               # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,                 # A top_k of 1 means the selected token is the most probable among all tokens.
    }
    chat = chat_model.start_chat(
        context=initial_context + " Message: " + response_text,  # 將兩者結合作為context
        examples=[
            InputOutputTextPair(
                input_text=' ',
                output_text=' ',
            ),
        ]
    )
    response = chat.send_message(message, **parameters)
    response_text = response.text  # 更新全域變數
    #print(f"Response from Model: {response_text}")
    return response_text

if __name__ == "__main__":
    initial_context = input("Enter the First initial context: ")  # 使用者輸入作為 initial_context
    while True:
        message = input("Enter the Message (or 'q' to quit): \n")
        if message == "q":
            print("End Chat")
            break
        response = science_tutoring(message, initial_context)
        print("-------------------Bot Message------------------------")
        print(f"\nBot response: {response}")
        print("-------------------Bot Message------------------------")
