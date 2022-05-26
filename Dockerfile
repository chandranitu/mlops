#Deriving the latest base image
#FROM python:3.7-slim
FROM python:3.8.5
RUN apt-get update
LABEL Maintainer="chandra"

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install -r requirements.txt

# Run the streamlit on container startup
CMD [ "streamlit", "run","--server.enableCORS","false","myapp.py" ]


#WORKDIR /home/hadoop123/data_ml
#COPY explore_data.py ./
#CMD [ "python", "./explore_data.py"]


# Run below command on console
#docker image build -t python:0.0.1 /home/hadoop123/data_ml
