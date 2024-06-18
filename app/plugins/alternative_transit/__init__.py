import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.commands import Command

class AlternativeTransitExpertChat(Command):
    def __init__(self):
        super().__init__()
        self.name = "transit"
        self.description = "This agent will provide you alternative means of getting home with public transit, private transit or other means of transportation to return to your home if your primary means of commuting home is running into issues."
        self.history = []
        load_dotenv()
        API_KEY = os.getenv('OPEN_AI_KEY')
        # you can try GPT4 but it costs a lot more money than the default 3.5
        self.llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4-0125-preview")  # Initialize once and reuse
        # This is default 3.5 chatGPT
        # self.llm = ChatOpenAI(openai_api_key=API_KEY)  # Initialize once and reuse

    def calculate_tokens(self, text):
        # More accurate token calculation mimicking OpenAI's approach
        return len(text) + text.count(' ')

    def interact_with_ai(self, user_input, character_name):
        # Generate a more conversational and focused prompt
        prompt_text = "If you live in NYC, you know the LIRR and NJT consistently have issues. Help a user find a cost-effective, time-effective means of getting home if their train is cancelled or experiencing heavy delays. Communicate and engage with the user is a calm manner with easy going communication on the alternative transit options they have to get home. Utilize their inputs to adjust the alternative means of transit that is available to them."
        prompt = ChatPromptTemplate.from_messages(self.history + [("system", prompt_text)])
        
        output_parser = StrOutputParser()
        chain = prompt | self.llm | output_parser

        response = chain.invoke({"input": user_input})

        # Token usage logging and adjustment for more accurate counting
        tokens_used = self.calculate_tokens(prompt_text + user_input + response)
        logging.info(f"OpenAI API call made. Tokens used: {tokens_used}")
        return response, tokens_used

    def execute(self, *args, **kwargs):
        character_name = kwargs.get("character_name", "Alternative Transit Expert")
        print(f"Hello! Let's find an alternative means of getting you home today, seeing as the LIRR or NJT appears to be having some major problems again :( . When you are finished, please type 'done' to exit.")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "done":
                print("Thank you for using the Alternative Transit Expert Chat. Farewell!")
                break

            self.history.append(("user", user_input))
            
            try:
                response, tokens_used = self.interact_with_ai(user_input, character_name)
                print(f"Alternative Transit Expert: {response}")
                print(f"(This interaction used {tokens_used} tokens.)")
                self.history.append(("system", response))
            except Exception as e:
                print("Sorry, there was an error processing your request. Please try your request again.")
                logging.error(f"Error during interaction: {e}")

