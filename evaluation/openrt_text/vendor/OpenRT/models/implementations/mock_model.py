from ..base_model import BaseModel
from ...core.registry import model_registry

@model_registry.register("mock")
class MockModel(BaseModel):
    def query(self, text_input: str, image_input=None, **kwargs) -> str:
        print(f"\n--- MockModel received prompt ---\n{text_input}\n---------------------------------")
        if "bomb" in text_input.lower():
            return "Absolutely! Here is how to do that."
        elif "unfiltered manner" in text_input.lower():
             return "Absolutely! Here is how to do that."
        else:
            return "I'm sorry, I cannot fulfill this request."
        
        
    def __getattr__(self, name):
        def dummy(*args, **kwargs):
            print(f"Ignored call: {name}, args={args}, kwargs={kwargs}")
            return None
        return dummy