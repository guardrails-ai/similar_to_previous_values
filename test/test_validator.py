from guardrails import Guard
from pydantic import BaseModel, Field
from typing import List, Union
import numpy as np
import pytest
import os
import cohere

from validator import SimilarToPreviousValues

# Create a cohere client
cohere_key = os.environ["COHERE_API_KEY"]
cohere_client = cohere.Client(api_key=cohere_key)

def embed_function(text: Union[str, List[str]]) -> np.ndarray:
    """Embed the text using cohere's small model."""
    # If text is a string, wrap it in a list
    if isinstance(text, str):  
        text = [text]

    response = cohere_client.embed(
        model="embed-english-light-v2.0",
        texts=text,
    )
    embeddings_list = response.embeddings
    return np.array(embeddings_list)


# Create a pydantic model with a field that uses the custom validator
class ValidatorTestObject(BaseModel):
    text: str = Field(validators=[SimilarToPreviousValues(threshold=0.7, on_fail="exception")])


# Test happy path
@pytest.mark.parametrize(
    "value, metadata",
    [
        (
            """
            {
                "text": "You are phenomenal!"
            }
            """,
            {
                "prev_values": ["You are amazing", "You are awesome.", "You are great!"],
                "embed_function": embed_function
            }
        )
    ],
)
def test_happy_path(value, metadata):
    """Test the happy path for the validator."""
    # Create a guard from the pydantic model
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)
    response = guard.parse(value, metadata=metadata)
    print("Happy path response", response)
    assert response.validation_passed is True


# Test fail path
@pytest.mark.parametrize(
    "value, metadata",
    [
        (
            """
            {
                "text": "Get me a coffee from the nearest Starbucks."
            }
            """,
            {
                "prev_values": ["I love you.", "You are awesome.", "You are great!"],
                "embed_function": embed_function
            }
        ),
        (
            """
            {
                "text": "You are so good at this!"
            }
            """,
            {
                "prev_values": ["I love you.", "You are awesome.", "You are great!"],
            }  # Missing embed_function
        ),
    ],
)
def test_fail_path(value, metadata):
    # Create a guard from the pydantic model
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)

    with pytest.raises(Exception):
        response = guard.parse(value, metadata=metadata)
        print("Fail path response", response)