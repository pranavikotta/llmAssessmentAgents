#using dspy modules to optimize chatbot prompts and measure metrics against different customer personas
import yaml
import dspy
from pydantic import BaseModel, Field
from typing import List, Optional

class SuggestedProduct(BaseModel):
    #mapping to the API Schema casing
    name: str = Field(description="ProductName from catalogue")
    provider: str = Field(description="Partner name / source")
    resolution: str = Field(description="Resolution with units (e.g. '15cm')")
    sensor_type: str = Field(description="Optical, SAR, Aerial, etc.")
    
    #ID management aligned
    offering_id: str = Field(description="The UUID AFTER 'oid:'")
    product_id: str = Field(description="The UUID AFTER 'pid:'")
    
    category: str = Field(description="Must be 'archive' or 'tasking'")
    image_url: str = Field(description="Sample image URL")
    
    #explanation fields for agent transparency
    reason: str = Field(description="Why this specifically fits the user's need")
    resolution_tier_advantage: str = Field(description="Benefit of this specific resolution range")

class RecommendationResponse(BaseModel):
    response: str = Field(description="Conversational response (No Markdown)")
    detected_location: Optional[str] = Field(default=None, alias="detectedLocation")
    needs_location_clarification: bool = Field(alias="needsLocationClarification")
    suggested_products: List[SuggestedProduct] = Field(alias="suggestedProducts")
    
    #guides for the frontend UI components
    resolution_guide: Optional[dict] = Field(default=None, alias="resolutionGuide")
    date_range_guide: Optional[dict] = Field(default=None, alias="dateRangeGuide")

    class Config:
        #allows the model to accept either camelCase or snake_case
        populate_by_name = True

with open("chatbot_system_prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

class GeneralQA(dspy.Signature):
    history = dspy.InputField(desc="Previous conversation history")
    question = dspy.InputField(desc="Customer's latest query")
    answer = dspy.OutputField(desc="Chatbot's response ending with a location prompt")

class Recommender(dspy.Signature):
    history = dspy.InputField(desc="Previous conversation history")
    question = dspy.InputField(desc="Customer's latest query")
    product_catalogue = dspy.InputField(desc="Available product catalogue data")
    recommendation_json = dspy.OutputField(desc="A valid JSON object containing response and suggested products")

#inject existing prompts into dspy signatures
GeneralQA.__doc__ = prompts["system_prompts"]["general_qa_prompt"]
Recommender.__doc__ = prompts["system_prompts"]["recommendations_prompt"]

class AIAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        #QA uses ChainOfThought, reasoning makes the expert answers better
        self.qa_bot = dspy.ChainOfThought(GeneralQA)
        
        #Recommender uses Predict to ensure it sticks to the JSON format
        self.rec_bot = dspy.Predict(Recommender)

    def forward(self, question, history, catalogue=None, location_detected=False):
        if location_detected:
            #Recommender Mode: Passes history, question, and the injected catalogue
            result = self.rec_bot(history=history, question=question, product_catalogue=catalogue)
            return result.recommendation_json
        
        #General QA Mode
        result = self.qa_bot(history=history, question=question)
        return result.answer, result.reasoning