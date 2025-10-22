# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Copy the simplified requirements file
COPY requirements.txt ./

# Install the dependencies
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy the new main.py file
COPY main.py ${LAMBDA_TASK_ROOT}/

# Set the command to run your function
# This tells Lambda to run the "handler" function inside the "main.py" file
CMD [ "main.handler" ]