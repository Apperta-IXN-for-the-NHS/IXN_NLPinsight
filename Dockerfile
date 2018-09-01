#########################################################################
# This code has been adapted from the following source:
# Link: https://docs.docker.com/samples/library/python/#create-a-dockerfile-in-your-python-app-project
#########################################################################

# Use an official Python runtime as a parent image
FROM python:3.6-stretch

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m spacy download en_core_web_sm


# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python","-u","main.py"]