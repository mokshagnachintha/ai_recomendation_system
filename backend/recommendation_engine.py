import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json

load_dotenv()

from products import PRODUCTS

class ProductRecommender:
    def __init__(self):
        # Ensure GOOGLE_API_KEY is in environment variables
        if not os.getenv("GOOGLE_API_KEY"):
            print("Warning: GOOGLE_API_KEY not found in environment variables.")
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        self.products_str = json.dumps(PRODUCTS, indent=2)

    def get_recommendations(self, user_preference: str):
        # [OPTION 2] Use this prompt to recommend REAL products from the internet (General Knowledge)
        # prompt_template = """
        # You are an expert product recommendation assistant.
        # Recommend the top 3 REAL available products from the market that match the user's preference: "{user_preference}".
        #
        # Return your response as a valid JSON array of objects. Each object should have:
        # - "id": A unique random number
        # - "name": The actual product name
        # - "price": Estimated price in USD (number only)
        # - "category": Product category
        # - "description": Brief description
        # - "reason": Why this specific product fits the preference
        #
        # Do not include any markdown formatting like ```json. Just return the raw JSON array.
        # """
        
        prompt_template = """
        You are an expert product recommendation assistant.
        
        Using the following list of available products:
        {products_data}
        
        Recommend the top 3 products that match the user's preference: "{user_preference}".
        
        Return your response as a valid JSON array of objects. Each object should have:
        - "id": The product ID
        - "name": The product name
        - "reason": A short explanation of why this fits the preference
        
        Do not include any markdown formatting like ```json. Just return the raw JSON array.
        """
        
        prompt = PromptTemplate(
            input_variables=["products_data", "user_preference"],
            template=prompt_template
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "products_data": self.products_str,
            "user_preference": user_preference
        })
        # Clean up potential markdown code blocks if the LLM insists on them
        cleaned_response = response.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)

