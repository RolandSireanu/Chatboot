FROM ubuntu:latest

# Set the working directory in the container
WORKDIR /app
EXPOSE 8000
COPY requirements.txt .

# Install Python (if not already included in the image)
RUN apt-get update && apt-get install -y python3 pip
RUN pip install --break-system-packages -r requirements.txt 

# # Copy the File.py from the local directory to the container
COPY server.py .
COPY chatboot.py .

# Start fastapi server with server.py file together with the uvicorn server

CMD ["python3", "server.py"]

