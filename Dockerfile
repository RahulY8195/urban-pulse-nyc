# Use an official Python runtime as a parent image
FROM python:3.14

# Set the working directory in the container
WORKDIR /app

# Copy the requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application scripts and preprocessed data
# Note: Ensure you have run preprocess_data.py first to generate preprocessed_data.csv
COPY generate_report.py /app/
COPY preprocessed_data.csv /app/

# Create output directory
RUN mkdir -p /app/output

# Run the generation script when the container launches
CMD ["python", "./generate_report.py"]
