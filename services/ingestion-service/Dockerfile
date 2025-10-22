# Use the official AWS Lambda Python base image for Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# Copy the requirements file
COPY requirements.txt ./

# --- ADD THIS BACK ---
# Install the basic 'gcc' compiler.
# This is NOT for numpy/sklearn, but for the 'hf-xet' package.
RUN yum install -y gcc gcc-c++

# Now, install the Python dependencies
# This will use wheels for numpy/sklearn (due to pinning)
# but will compile hf-xet (using the gcc we just installed)
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy all your project files (handler.py, rss_feeds.py, etc.)
COPY . ${LAMBDA_TASK_ROOT}/

# Set the command to run your function
CMD [ "handler.ingest" ]