FROM node:18.16.1

# Copy requirements.txt and install pip requirements
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Set the working directory in the container
WORKDIR /app

# Copy the application files to the working directory
COPY . /app

# Command to run the application
CMD ["python3", "app.py"]
