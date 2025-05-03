# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies (e.g., from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Install Git (if needed) and other tools
RUN apt-get update && apt-get install -y git

# Set up the default command to run when the container starts
CMD ["bash"]
