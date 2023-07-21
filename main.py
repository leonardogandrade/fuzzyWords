"""
    Main Module docstring
"""


from fastapi import FastAPI
from routes.text_check import sentence
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

origins = {
    'http://localhost:3000',
    'http://localhost',
    'http://localhost/textcheck'
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentence)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8001)
