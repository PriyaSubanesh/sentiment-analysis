# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "vaderSentiment",
# ]
# ///

from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Create the FastAPI application.
app = FastAPI(
    title="Batch Sentiment Analysis API",
    description="Classifies sentences as happy, sad, or neutral.",
    version="1.0.0",
)

# Create the sentiment analyser once when the server starts.
analyser = SentimentIntensityAnalyzer()

# A sentence must be a non-empty string.
Sentence = Annotated[str, Field(min_length=1, max_length=2000)]


# This describes the JSON body that the API accepts.
class SentimentRequest(BaseModel):
    sentences: list[Sentence] = Field(
        min_length=1,
        max_length=1000,
        description="A non-empty list of sentences to analyse.",
    )

    # Reject strings that contain only spaces, such as "   ".
    @field_validator("sentences")
    @classmethod
    def sentences_must_not_be_blank(cls, sentences: list[str]) -> list[str]:
        for sentence in sentences:
            if not sentence.strip():
                raise ValueError("Each sentence must contain text, not only spaces.")
        return sentences


# This describes one item in the returned results list.
class SentimentResult(BaseModel):
    sentence: str
    sentiment: Literal["happy", "sad", "neutral"]


# This describes the complete JSON response.
class SentimentResponse(BaseModel):
    results: list[SentimentResult]


def get_sentiment(sentence: str) -> Literal["happy", "sad", "neutral"]:
    """
    Convert VADER's sentiment score into one of the three labels
    required by the assignment.
    """
    scores = analyser.polarity_scores(sentence)
    compound_score = scores["compound"]

    # VADER convention:
    # compound >= 0.05: positive
    # compound <= -0.05: negative
    # otherwise: neutral
    if compound_score >= 0.05:
        return "happy"

    if compound_score <= -0.05:
        return "sad"

    return "neutral"


@app.get("/")
async def root():
    return {
        "message": "Batch Sentiment Analysis API is running.",
        "docs": "/docs",
    }


@app.post("/sentiment", response_model=SentimentResponse)
async def analyse_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    Analyse every sentence and return results in exactly the same order.
    """
    if not request.sentences:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one sentence.",
        )

    results = [
        SentimentResult(
            sentence=sentence,
            sentiment=get_sentiment(sentence),
        )
        for sentence in request.sentences
    ]

    return SentimentResponse(results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)