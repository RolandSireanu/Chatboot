FROM ubuntu:latest

# Set the working directory in the container
WORKDIR /app
COPY app/ .

# Install Python (if not already included in the image)
RUN apt-get update && apt-get install -y python3 pip
RUN pip install -r requirements.txt --break-system-packages

# Command to run the Python script when the container starts
CMD ["python3", "chatboot.py"]
