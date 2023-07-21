"""
    Sentence Module docstring
"""


from fastapi import APIRouter, Body, status
from fastapi.responses import JSONResponse
from classes.nlp import UniversalSentenceEncoder

sentence = APIRouter()

sentence_encoder = UniversalSentenceEncoder()
sentence_encoder.loadModel()

@sentence.post('/textcheck')
async def compare_sentences(threshold:float, body = Body(...)):

    with open('db/db.txt', 'r+') as projects:
        doc = projects.read().splitlines()
    
    query = body['query']

    result = sentence_encoder.sentence_similarity(doc, query, threshold)
    # result = {'msg': 'hello'}
    return JSONResponse(content=result, status_code=status.HTTP_200_OK)

@sentence.get('/sentences')
async def hello():
    return JSONResponse(content={"msg": "hello"}, status_code=status.HTTP_200_OK)
