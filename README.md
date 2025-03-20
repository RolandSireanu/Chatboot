# Chatboot
Chatboot is an interactive chatbot designed on top of RAG and GPT-3.5 model, having access to all the PFD file provided into resources directory.
By using a docker container as it's main deployment environment, it facilitates easy integration into various applications and platforms.

As a prerequisite for this project you need an OPEN_API_KEY set in your env variables.

Build docker image:
`docker build -t <img_name> .`

Run docker container:
`docker run -it --rm -e OPENAI_API_KEY=$OPENAI_API_KEY <img_name>`