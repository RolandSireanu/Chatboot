# Chatboot
Chatboot is an interactive chatbot designed on top of RAG and GPT-3.5 model, having access to all the PFD file provided into resources directory.
By using a docker container as it's main deployment environment, it facilitates easy integration into various applications and platforms.

As a prerequisite for this project you need an OPEN_API_KEY set in your env variables.

Build docker image:
docker-compose build

Run docker container:
docker-compose up

In order to interact with the chatboot you can use client/client.py application.
Both sides are using websocket for data communication and have the following API:
1. Upload a pdf file to server (as input for RAG)
"/upload path_to_file"

2. Address a question
"question to address"

