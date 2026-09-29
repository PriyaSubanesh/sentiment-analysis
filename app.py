from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Create the FastAPI application.
app = FastAPI(
    title="Batch Sentiment Analysis API",
    description="Classifies sentences as happy, sad, or neutral."
)


# Allow the assignment evaluator, browsers, and other websites
# to send requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the sentiment analyser once when the API starts.
analyser = SentimentIntensityAnalyzer()


# This describes the JSON that the API accepts.
class SentimentRequest(BaseModel):
    sentences: list[str] = Field(
        min_length=1,
        description="A list containing one or more sentences."
    )


# This describes one result inside the results list.
class SentimentResult(BaseModel):
    sentence: str
    sentiment: Literal["happy", "sad", "neutral"]


# This describes the JSON that the API returns.
class SentimentResponse(BaseModel):
    results: list[SentimentResult]


def get_sentiment(sentence: str) -> Literal["happy", "sad", "neutral"]:
    """
    Turn VADER's numerical sentiment score into one of the
    three labels required by the assignment.
    """
    score = analyser.polarity_scores(sentence)["compound"]

    if score >= 0.05:
        return "happy"

    if score <= -0.05:
        return "sad"

    return "neutral"


# A simple route to confirm that the API is running.
@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API is running.",
        "endpoint": "/sentiment",
        "docs": "/docs"
    }


# The assignment endpoint.
@app.post("/sentiment", response_model=SentimentResponse)
async def analyse_sentiment(data: SentimentRequest) -> SentimentResponse:
    results = [
        SentimentResult(
            sentence=sentence,
            sentiment=get_sentiment(sentence)
        )
        for sentence in data.sentences
    ]

    return SentimentResponse(results=results)
